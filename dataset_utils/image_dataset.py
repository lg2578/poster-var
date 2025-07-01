import os
import cv2
import torch
from PIL import Image
import torch
import pandas as pd
from dataset_utils.transforms import get_data_transforms
class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 root: str,            #数据集根目录
                 transform=None,
                 data_size:int=384,
                 cache_img=False,#cache到内存，可以提速一倍
                 csv=None, #如果指定了csv文件就从csv文件读取训练集数据，否则从目录中读取
                 folds=[], #根据csv文件中的fold，筛选folds指定的fold
                 frac=1,   #从数据集中采样比例，比如测试集太大的时候，可以用一个小的测试集进行测试
                 sample_numbers=0,#如果=0，则不对训练集中的样本进行平衡，否则按类别=sample_numbers进行样本平衡
                 votes_sum=0
                 ):
        # notice that:
        # sub_data_size mean sub-image's width and height.
        """ basic information """
        self.root = root
        self.transform=transform
        self.count=0
        self.emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
        self.votes_sum=votes_sum

        self.root=root
        self.sample_numbers=sample_numbers
        self.cache_img=cache_img
        self.folds=folds


        """ read all data information """
        if csv!=None:
            self.data_infos=self.getDataInfo_csv(root,csv,folds,frac)
        else:
            self.data_infos = self.getDataInfo_fold(root)

    def getDataInfo_fold(self, root):
        data_infos = []
        folders = os.listdir(root)
        folders.sort() # sort by alphabet
        for class_id, folder in enumerate(folders):
            files = os.listdir(root+folder)
            for file in files:
                data_path = root+folder+"/"+file
                data_infos.append({"path":data_path, "label":class_id})
        return data_infos
    def getDataInfo_csv(self,root, csv,folds,frac):
        df=pd.read_csv(csv)
        df=df[df['fold'].isin(folds)]
        if frac!=1:
            df=df.sample(frac=frac)
        self.df=df

        data_infos = []
        for row in df.itertuples():
            data_path = os.path.join(root,row.path)
            # 取出 neutral 到 contempt 的 8 个字段
            if self.votes_sum>0:
                values = [getattr(row, emotion) for emotion in self.emotions]
               #如果votes不相等，则有错误，或者是没提供详细的votes，后面会根据label生成onehot形式的label_em
                normalized_list = [v / self.votes_sum for v in values]
            else:
                normalized_list=[]
            normalized_tensor = torch.tensor(normalized_list, dtype=torch.float32)
            data_infos.append({"path":data_path, "label":row.label,"label_em":normalized_tensor})
        del df
        return data_infos
    def class_balance_resample(self, numbers):  
        """
        对每个类别随机采样 numbers 个样本，用于多分类样本平衡。
        如果某个类别样本数量小于 numbers，则保留全部。
        """
        df = self.df
        data_infos = []
        class_labels = df['label'].unique()
        
        sampled_df_list = []

        for label in class_labels:
            df_class = df[df['label'] == label]
            if len(df_class) > numbers:
                df_sampled = df_class.sample(n=numbers, random_state=self.seed) if hasattr(self, 'seed') else df_class.sample(n=numbers)
            # 如果不足，保留全部
            else:
                # 有放回采样（重复采样）补足到 numbers 个
                df_sampled = df_class.sample(n=numbers, replace=True, random_state=self.seed) if hasattr(self, 'seed') else df_class.sample(n=numbers, replace=True)
        
            sampled_df_list.append(df_sampled)

        # 合并所有类别并打乱
        df_sampled = pd.concat(sampled_df_list).sample(frac=1, random_state=self.seed if hasattr(self, 'seed') else None).reset_index(drop=True)

        print(f"[dataset] number of resamples: {len(df_sampled)}")

        for row in df_sampled.itertuples():
            data_path = os.path.join(self.root, row.path)
            data_infos.append({"path": data_path, "label": row.label})
        
        self.data_infos = data_infos

    def class_balance_resample2(self, numbers):  
        """
        对每个类别随机采样 numbers 个样本，用于多分类样本平衡。
        如果某个类别样本数量小于 numbers，则保留全部。
        """
        df = self.df
        data_infos = []
        class_labels = df['label'].unique()
        
        sampled_df_list = []

        for label in class_labels:
            df_class = df[df['label'] == label]
            if len(df_class) > numbers:
                df_class = df_class.sample(n=numbers, random_state=self.seed) if hasattr(self, 'seed') else df_class.sample(n=numbers)
            # 如果不足，保留全部
            sampled_df_list.append(df_class)

        # 合并所有类别并打乱
        df_sampled = pd.concat(sampled_df_list).sample(frac=1).reset_index(drop=True)

        print(f"[dataset] number of resamples: {len(df_sampled)}")

        for row in df_sampled.itertuples():
            data_path = os.path.join(self.root, row.path)
            data_infos.append({"path": data_path, "label": row.label})
        
        self.data_infos = data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        # get data information.
        self.count+=1

        image_path = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]
        label_em=self.data_infos[index]["label_em"]
        # read image by opencv.如果已经加载，就从内存中读取
        if self.cache_img==True and 'img' in self.data_infos[index]:
            img=self.data_infos[index]["img"]   
        else: 
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"图像读取失败: {image_path}")
            img = img[:, :, ::-1] # BGR to RGB.
            # to PIL.Image
            img = Image.fromarray(img)
            if self.cache_img==True:
                self.data_infos[index]["img"]=img

        if self.transform!=None:
            img = self.transform(img)

        return index, img, label,label_em,image_path
                   
def build_img_dataloader(args):

    data_transforms=get_data_transforms(args)
    train_set, train_loader = None, None
    if args.train_root is not None:
        train_set = ImageDataset(root=args.train_root, transform=data_transforms['train'],data_size=args.train_size,cache_img=args.cache_img,
                                csv=args.metadata,folds=args.train_folds,frac=args.train_frac,sample_numbers=args.sample_numbers,votes_sum=args.votes_sum)
        train_loader = torch.utils.data.DataLoader(train_set, num_workers=args.num_workers, shuffle=args.train_shuffle, batch_size=args.batch_size,
                                                   persistent_workers=True)

    val_set, val_loader = None, None
    if args.val_root is not None:
        val_set = ImageDataset(root=args.val_root, transform=data_transforms['valid'],data_size=args.val_size,cache_img=args.cache_img,
                            csv=args.metadata,folds=args.val_folds,frac=args.val_frac,votes_sum=args.votes_sum)
        val_loader = torch.utils.data.DataLoader(val_set, num_workers=args.num_workers, shuffle=False, batch_size=args.batch_size*2,
                                                 persistent_workers=True)

    return train_loader, val_loader



