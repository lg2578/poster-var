import torch.optim.adamw
import math
import numpy as np 
from collections import Counter
from torch.optim.lr_scheduler import LRScheduler,ExponentialLR,CosineAnnealingLR
import swanlab
class BrachistochroneLRScheduler(LRScheduler):
    def __init__(self,optimizer, init_lr, max_epochs,max_batchs,last_epoch=-1, verbose="deprecated"):
        """
        初始化最速降线学习率调度器
        :param initial_lr: 初始学习率
        :param max_epochs: 最大训练轮数
        """
        self.max_batchs=max_batchs
        self.last_step=0
        self.schedule,self.counter=self.brachistochrone(max_epochs,max_batchs,init_lr)
        super(BrachistochroneLRScheduler, self).__init__(optimizer, last_epoch,verbose)
    def brachistochrone(self,epochs,batchs,init_lr):
        # 定义参数范围
        steps=epochs*batchs
        theta = np.linspace(0, np.pi, steps)
        x = theta - np.sin(theta)
        y = 1 + np.cos(theta)
        x=x/np.pi *epochs
        x=np.round(x).astype(int)
        init_lr=0.001
        y=y/2*init_lr
        counter = Counter(x)
        return y,counter

    def get_lr(self):
        """
        根据当前轮数获取学习率
        :param epoch: 当前训练轮数
        :return: 调整后的学习率
        """
        #self.last_epoch是每个step加1，如果在batch里调用，就是每个batch加1
        epoch=self.last_epoch//self.max_batchs
        batch=self.last_epoch%self.max_batchs
        if batch==0:
            step_length=1
        else:
            step_length=self.counter[epoch]/self.max_batchs
        self.last_step=self.last_step+step_length
        self.last_step=self.last_step
        lr=self.schedule[int(np.round(self.last_step))]
        # print(self.last_epoch,self.last_step)
        return [lr]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        if param_group["lr"] is not None:
            return param_group["lr"]

def adjust_lr(wandb,schedule,epoch,batch_id,batchs):
    #每个batch更新的学习率调度函数
    step_lr=[]
    if type(schedule) in step_lr:
        # writer.add_scalar('lr_epoch/train',get_lr(schedule.optimizer),epoch)
        wandb.log({'step/train':epoch*batchs+batch_id,'lr_step/train':get_lr(schedule.optimizer)})
        swanlab.log({'step/train':epoch*batchs+batch_id,'lr_step/train':get_lr(schedule.optimizer)})
        #t_max不能等于0，小于0，则在最大，最小值间摆动，这里设置t_max<=0,则不调整学习率
        schedule.step()
    else:
        if (batch_id+1)==batchs:
            # writer.add_scalar('lr_epoch/train',get_lr(schedule.optimizer),epoch)
            wandb.log({'epoch':epoch,'lr_epoch/train':get_lr(schedule.optimizer)})
            swanlab.log({'epoch':epoch,'lr_epoch/train':get_lr(schedule.optimizer)})
            #t_max不能等于0，小于0，则在最大，最小值间摆动，这里设置t_max<=0,则不调整学习率
            if type(schedule)==CosineAnnealingLR and  schedule.T_max==-1:
                pass
            else:
                schedule.step()



def cosine_decay(args, batchs: int, decay_type: int = 1):
    total_batchs = args.max_epochs * batchs
    iters = np.arange(total_batchs - args.warmup_batchs)

    if decay_type == 1:
        schedule = np.array([1e-12 + 0.5 * (args.init_lr - 1e-12) * (1 + \
                             math.cos(math.pi * t / total_batchs)) for t in iters])
    elif decay_type == 2:
        schedule = args.init_lr * np.array([math.cos(7*math.pi*t / (16*total_batchs)) for t in iters])
    else:
        raise ValueError("Not support this deccay type")
    
    if args.warmup_batchs > 0:
        warmup_lr_schedule = np.linspace(1e-9, args.init_lr, args.warmup_batchs)
        schedule = np.concatenate((warmup_lr_schedule, schedule))

    return schedule

if __name__=='__main__':
    import torch
    import argparse
    from configs import load_yaml
    import matplotlib.pyplot as plt
    #获取命令行配置文件
    parser = argparse.ArgumentParser("Fine-Grained Visual Classification")
    parser.add_argument("--c", default="", type=str, help="config file path")
    args = parser.parse_args()
    assert args.c != "", "Please provide config file (.yaml)"
    load_yaml(args, args.c)
    model=torch.nn.Linear(10,200)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0001)
    # lr_sche=WarmupCosineAnnealingLR(optimizer,2,args.max_epochs,eta_min=0)
    # lr_sche=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,50)
    lr_sche=torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=0.01,total_iters=args.warmup_batchs)
    args.max_epochs=50
    args.warmup_batchs=200
    # lr_sche=cosine_decay(args,32)
    x=[]
    for i in range(args.max_epochs*32):
        # print(f'{i}:',lr_sche.get_last_lr())
        x.append(lr_sche.get_last_lr())
        # x.append(get_lr(optimizer))
        optimizer.step()
        # adjust_lr(i,optimizer,lr_sche)
        lr_sche.step()
    plt.plot(x)
    plt.show()
       