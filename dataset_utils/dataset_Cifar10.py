import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader

class CIFAR10ClassMapper:
    def __init__(self):
        # 定义 CIFAR - 10 数据集的类别名称列表
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']

    def get_class_name(self, class_index):

        if 0 <= class_index < len(self.class_names):
            return self.class_names[class_index]
        return None
class ReturnIndexCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train = True, transform = None, target_transform = None, download = False):
        super().__init__(root, train, transform, target_transform, download)
         #修正一些错误标定,但验证集正确率还降低了！
        self.modify_labels={}

    def __getitem__(self, index):
        # 调用父类的 __getitem__ 方法获取样本和标签
        img, label = super().__getitem__(index)

        return index,img, label
def build_dataloader(args):
    
    train_size=args.train_size
    val_size=args.val_size
    #先放大一点再crop
    resize_train=round(train_size*1.3)
    resize_val=round(val_size*1.3)
    """ declare data augmentation """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
 
    train_transforms = transforms.Compose([
        
                transforms.Resize((resize_train, resize_train), Image.BILINEAR),
                transforms.RandomCrop((train_size, train_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                transforms.ToTensor(),
                normalize
        ])

    valid_transforms = transforms.Compose([
         
                transforms.Resize((resize_val, resize_val), Image.BILINEAR),
                transforms.CenterCrop((val_size, val_size)),
                transforms.ToTensor(),
                normalize
        ])

    train_dataset = ReturnIndexCIFAR10(
        root=args.train_root,  # 数据存储的根目录
        train=args.train_shuffle,     # 是否为训练集
        download=True,  # 是否下载数据集
        transform=train_transforms  # 应用数据预处理
    )

    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset,  # 训练数据集
        batch_size=args.batch_size,  # 每个批次的数据样本数量
        shuffle=True ,   # 是否在每个 epoch 开始时打乱数据
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    # 下载并加载测试集
    valid_dataset = ReturnIndexCIFAR10(
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

