#20250202,args.batch_size只能是1
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import timm
import argparse
from utils.configs import load_yaml
from dataset_utils.builder import build_dataloader
from utils.configs import seed_everything
from utils.visualization import tensor2img
from trials.models import create_model
import os
seed_everything(0)
parser = argparse.ArgumentParser("Fine-Grained Visual Classification")
parser.add_argument("--c", default="", type=str, help="config file path")
args = parser.parse_args()
assert args.c != "", "Please provide config file (.yaml)"
cfg_file = args.c.split(',')[0]
cfg_path=os.path.join('./configs/',cfg_file)
load_yaml(args, cfg_path)
args.batch_size=1
args.train_shuffle=False
# args.pretrained_local=f'./logs/{args.dataset_name}/{args.model_name}/2025-01-28-14-19/best.pt'
checkpoint = torch.load(args.pretrained_local, map_location=torch.device('cpu'),
                        weights_only=True)
pretrained_models=['ir50','POSTERv2','ResEmoteNet']
if args.model_name in pretrained_models:
    model=create_model(args)
else:
    model = timm.create_model(f'hf_hub:{args.model_name}',pretrained=args.pretrained_timm,num_classes=args.num_classes)

model.load_state_dict(checkpoint['model'])
device=torch.device('cuda')
model=model.to(device)
model.eval()

# _,val_loader=build_dataloader(args)
#在训练集上看看效果
_,val_loader=build_dataloader(args)
val_loader.num_workers=0
err_count=0
sample_count=0
# print(model)
if 'convnextv2' in args.model_name:
    target_layers = [model.stages[-1]]
if 'resnet' in args.model_name:
    target_layers = [model.layer4[-1]]
if 'POSTERv2' in args.model_name:
    target_layers = [model.conv3]# face_landback,conv3
# mapper=CIFAR10ClassMapper()
for idx,(idxs, datas,labels,labels_em,image_path) in enumerate(val_loader.dataset):
    labels=torch.tensor([labels])
    datas=datas.unsqueeze(0)
    sample_count+=len(labels)
    datas=datas.to(device)
    labels=labels.to(device)
    # if 'test_2393_aligned' not in image_path:
    #     continue

    with torch.no_grad():
        outs=model(datas)
        outs2=torch.softmax(outs,1)
        outs=torch.argmax(outs,1)
    #!=改成==，可以显示正确分类的图像热图信息
    err_idxs=outs!=labels
    #对于特定索引的样本，强制将其标记为预测错误
    # if idx in [7000,10994,35310,40144]:
    #     err_idxs=[True]
    err_outs=outs[err_idxs]
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统
    if len(err_outs>0):
        err_count+=len(err_outs)
        err_datas=datas[err_idxs]

        print(f'{idxs}:{labels[err_idxs].cpu().numpy()}->{err_outs.cpu().numpy()}')
        input_tensor =err_datas[0] # Create an input tensor image for your model.
        img_input = tensor2img(input_tensor)
        input_tensor=input_tensor.unsqueeze(0)
        # Note: input_tensor can be a batch tensor with several images!
        # We have to specify the target we want to generate the CAM for.
        err_targets = [ClassifierOutputTarget(err_outs[0].cpu().numpy())]
        # img_input2=np.zeros_like(img_input)

        # Construct the CAM object once, and then re-use it on many images.
        with GradCAM(model=model, target_layers=target_layers) as cam:
            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            grayscale_cam = cam(input_tensor=input_tensor, targets=err_targets)
            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam = grayscale_cam[0, :]
            cam_err = show_cam_on_image(img_input, grayscale_cam, use_rgb=True)
            # You can also get the model outputs without having to redo inference
            # model_outputs = cam.outputs

        lbl_targets = [ClassifierOutputTarget(labels[0].cpu().numpy())]

        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=lbl_targets)
            grayscale_cam = grayscale_cam[0, :]
            cam_lbl = show_cam_on_image(img_input, grayscale_cam, use_rgb=True)
  
        transforms=val_loader.dataset.transform
        val_loader.dataset.transform=torchvision.transforms.ToTensor()
        img_ori=val_loader.dataset[idx][1]
        val_loader.dataset.transform=transforms
        img_ori =tensor2img(img_ori)

        imgs_info=[('类激活图-标定类',cam_lbl),('类激活图-预测类',cam_err),
                   ('数据增强',img_input),('原始图',img_ori)]
        
        fig, axes = plt.subplots(1, 4, figsize=(10, 5)) 
        for i,(k,v) in enumerate(imgs_info):
            axes[i].imshow(v)
            axes[i].set_title(k)
            axes[i].axis('off')

        # plt.show()
        # plt.close()
        # break
print(err_count,sample_count)