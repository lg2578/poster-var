
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torch
from torch.utils.data import DataLoader

class ReturnIndexMNIST(torchvision.datasets.MNIST):
    def __init__(self, root, train = True, transform = None, target_transform = None, download = False):
        super().__init__(root, train, transform, target_transform, download)
         #修正一些错误标定,但验证集正确率还降低了！
        self.modify_labels={10994:9,35310:6,40144:3,43454:3,53396:4,57744:4,59915:7}
        # self.modify_labels={}

    def __getitem__(self, index):
        # 调用父类的 __getitem__ 方法获取样本和标签
        img, label = super().__getitem__(index)
        # 修正错误标定
        # if index in self.modify_labels:
            # label=self.modify_labels[index]
        return index,img, label

def build_dataloader(args):
    
    train_size=args.train_size
    val_size=args.val_size
    #先放大一点再crop
    resize_train=round(train_size*1.3)
    resize_val=round(val_size*1.3)
    """ declare data augmentation """
    normalize = transforms.Normalize((0.1307,), (0.3081,))
 
    train_transforms = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((resize_train, resize_train), Image.BILINEAR),
                transforms.RandomCrop((train_size, train_size)),
                # transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                transforms.ToTensor(),
                normalize
        ])

    valid_transforms = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((resize_val, resize_val), Image.BILINEAR),
                transforms.CenterCrop((val_size, val_size)),
                transforms.ToTensor(),
                normalize
        ])

    train_dataset = ReturnIndexMNIST(
        root=args.train_root,  # 数据存储的根目录
        train=True,     # 是否为训练集
        download=True,  # 是否下载数据集
        transform=train_transforms  # 应用数据预处理
    )
   

    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset,  # 训练数据集
        batch_size=args.batch_size,  # 每个批次的数据样本数量
        shuffle=args.train_shuffle ,   # 是否在每个 epoch 开始时打乱数据
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    # 下载并加载测试集
    valid_dataset = ReturnIndexMNIST(
        root=args.val_root,
        train=False,    # 是否为测试集
        download=True,
        transform=valid_transforms
    )

    # 创建测试数据加载器
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size*2,
        shuffle=False,   # 测试集通常不需要打乱数据
        num_workers=args.num_workers,
        persistent_workers=True,
    )



    return train_loader, valid_loader

