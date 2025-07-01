import torch
import contextlib
import warnings
import os
import shutil
from dataset_utils.builder import build_dataloader
from utils.configs import load_yaml
from utils.configs import seed_everything
from torch.utils.tensorboard import SummaryWriter
warnings.simplefilter("ignore")
from datetime import datetime
import argparse
import timm
from trials.sam import SAM

from train_encoder import trainloop
from valid_encoder import validloop
from train_encoder_sam import trainloop as trainloop_sam
import swanlab
import wandb
import torch.nn.functional as F
from trials.models import create_model

def init_environment(args):
    seed_everything(args.seed)
    ### = = = =  Model = = = =  
    print(f"Building Model....{args.model_name},pretrained_timm={args.pretrained_timm},pretrained_local={args.pretrained_local}")
    pretrained_models=['ir50','POSTERv2','ResEmoteNet']
    if args.model_name in pretrained_models:
        model=create_model(args)
    else:
        model = timm.create_model(f'hf_hub:{args.model_name}',pretrained=args.pretrained_timm,num_classes=args.num_classes)
    #pretrained控制是否从timm加载权重,也可以从本地加载,重新训练start_epoch=0,继续训练start_epoch=last.pt中保存的值
    if args.pretrained_local is not None:
        checkpoint = torch.load(args.pretrained_local, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    #测试改进模型
    # model=test_model(model,args.num_classes)
    #多卡训练
    # model = torch.nn.DataParallel(model, device_ids=None) # device_ids : None --> use all gpus.
    model.to(args.device)

    ### = = = =  Dataset and Data Loader = = = =  
    print(f"Building Dataloader....{args.dataset_name},cache_img={args.cache_img},sample_numbers:{args.sample_numbers}")
    
    train_loader, val_loader = build_dataloader(args)
    
    if train_loader is None and val_loader is None:
        raise ValueError("Find nothing to train or evaluate.")

    if train_loader is not None:
        print(f"Train Samples: {len(train_loader.dataset)},folds={args.train_folds},"
              f"batchs: {len(train_loader)},frac={args.train_frac},data_size={args.train_size}")
    else:
        # raise ValueError("Build train loader fail, please provide legal path.")
        print("Train Samples: 0 ~~~~~> [Only Evaluation]")
    if val_loader is not None:
        print(f"Validation Samples: {len(val_loader.dataset)},folds={args.val_folds},"
             f"batchs: {len(val_loader)},frac={args.val_frac},data_size={args.val_size}")

    else:
        print("Validation Samples: 0 ~~~~~> [Only Training]")
    
    if train_loader is None:
        return train_loader, val_loader, model, None, None, None, None,None
    
    ### = = = =  Optimizer = = = =  
    print(f"Building Optimizer....{args.optimizer},init_lr={args.init_lr},t_max={args.t_max},max_epochs={args.max_epochs}")
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr, nesterov=True, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr,weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr,weight_decay=args.weight_decay)
    else:
        ValueError("未定义的优化器")
    print(f'{args.use_sam=},{args.schedule_name=},{args.use_amp=}')
    if args.use_sam:
        optimizer = SAM(model.parameters(), type(optimizer), lr=args.init_lr, rho=args.rho,adaptive=False,weight_decay=args.weight_decay )
    
    if args.schedule_name =='CosineAnnealingLR':
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.t_max)
    elif args.schedule_name =='ExponentialLR':
        schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    if args.pretrained_timm==False and args.pretrained_local is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if args.use_amp:
        scaler = torch.amp.GradScaler()
        amp_context = torch.amp.autocast
    else:
        scaler = None
        amp_context = contextlib.nullcontext

    return train_loader, val_loader, model, optimizer, schedule, scaler, amp_context, start_epoch


def main(args, wandb):
    """
    save model last.pt and best.pt
    """
    train_loader, val_loader, model, optimizer, schedule, scaler, amp_context, start_epoch = init_environment(args)
    print("loading embedding pt file")
    if args.embedding_file!=None:
        embeddings=torch.load(args.embedding_file,map_location=torch.device('cpu'))
    else:
        # 进行 one - hot 编码
        numbers = torch.arange(args.num_classes)
        one_hot_encoded = F.one_hot(numbers)
        embeddings=one_hot_encoded.float()
    embeddings=embeddings.to(args.device)
    best_metric_value = torch.tensor(0.0) if args.metric_direction=='up' else torch.tensor(1000.0)

    if train_loader is None:
        epochrange=range(1)
    else:
        epochrange=range(start_epoch, args.max_epochs)
    for epoch in epochrange:
        #train
        if train_loader is not None:
            if args.sample_numbers>0:
                train_loader.dataset.class_balance_resample(args.sample_numbers)
            if args.use_sam:
                trainloop_sam(wandb,args, epoch, model, scaler, amp_context, optimizer,
                        schedule, train_loader,embeddings)
            else:
                trainloop(wandb,args, epoch, model, scaler, amp_context, optimizer,
                        schedule, train_loader,embeddings)
            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch":epoch}
            torch.save(checkpoint, os.path.join(args.logdir,"last.pt"))

        #valid
        if  (epoch + 1) % args.eval_freq == 0 and val_loader is not None: 
            metric_value =validloop(wandb,args, model,epoch, val_loader,embeddings)
            save_model=(args.metric_direction=='up' and (metric_value > best_metric_value)) \
                    or (args.metric_direction=='down' and (metric_value < best_metric_value))

            if save_model:
                best_metric_value = metric_value
                try:
                    torch.save(checkpoint, os.path.join(args.logdir,"best.pt"))
                except:
                    print('best.pt 未保存')#valid模式，没有进行训练，没有checkpoint
            print(f"best_{args.metric_name}:{best_metric_value:.4f},{args.metric_name}:{metric_value:.4f}")
            wandb.log({'best_metric_value':best_metric_value,'epoch':epoch})
            swanlab.log({'best_metric_value':best_metric_value,'epoch':epoch})
    #手动释放内存和显存
    train_loader,val_loader,model,optimizer=None,None,None,None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":

    #获取命令行配置文件
    parser = argparse.ArgumentParser("Fine-Grained Visual Classification")
    parser.add_argument("--c", default="", type=str, help="config file path")
    args = parser.parse_args()
    assert args.c != "", "Please provide config file (.yaml)"
    log_root=r'd:lg\logs'
    for cfg_file in args.c.split(','):
        cfg_path=os.path.join('./configs/',cfg_file)
        load_yaml(args,cfg_path )
        log_time = "{0:%Y-%m-%d-%H-%M}".format(datetime.now())
        logdir=os.path.join(log_root,args.dataset_name,args.model_name,log_time)
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device=torch.device('cpu')
        args.device=device
        args.logdir=logdir
        args.logtime=log_time
        #copy config files to logdir
        cfg_name=os.path.basename(args.c)
        transforms_name='transforms.py'
        os.makedirs(logdir)
        shutil.copyfile(cfg_path,os.path.join(logdir,cfg_name))
        shutil.copyfile('./dataset_utils/'+transforms_name,os.path.join(logdir,transforms_name))
        swanlab.sync_wandb()
        wandb.init(project=f'FGVC-{args.dataset_name}-{args.num_classes}',name=log_time,config=args,dir=log_root,mode='offline')
        main(args,wandb)
        wandb.finish()