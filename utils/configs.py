import yaml
import os
import argparse
import random
import numpy as np
import torch

def load_yaml(args, yml):
    with open(yml, 'r', encoding='utf-8') as fyml:
        dict = yaml.load(fyml.read(), Loader=yaml.Loader)
        for k in dict:
            setattr(args, k, dict[k])
        # args.dict=dict
        # args.dict['train_folds']=','.join(map(str,args.dict['train_folds']))
        # args.dict['val_folds']=' '.join(map(str,args.dict['val_folds']))
        
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


