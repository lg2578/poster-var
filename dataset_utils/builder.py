
from dataset_utils.image_dataset import build_img_dataloader

from dataset_utils import dataset_MNIST
from dataset_utils import dataset_Cifar10


def build_dataloader(args):
    train_loader,val_loader=None,None
    if args.dataset_name in['CUB200-2011','ISIC2024','SCUTv2','AffectNet','RAFDB','AffectNet_RAFDB','FER2013']:
        train_loader, val_loader=build_img_dataloader(args)

    if args.dataset_name in ['MNIST']:
        train_loader,val_loader=dataset_MNIST.build_dataloader(args)
    if args.dataset_name in ['Cifar10']:
        train_loader,val_loader=dataset_Cifar10.build_dataloader(args)

    return train_loader, val_loader
        
        

