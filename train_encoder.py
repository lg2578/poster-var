import warnings
from tqdm import tqdm
from utils.lr_schedule import cosine_decay, adjust_lr, get_lr
from utils.metrics import Metrics
from loss_encoder import compute_loss_em
import torch
import swanlab

warnings.simplefilter("ignore")
def trainloop(wandb,args, epoch, model, scaler, amp_context, optimizer, schedule,train_loader,embeddings):
    batchs=len(train_loader)
    #trainloop只度量主要指标
    metrics=Metrics(args.num_classes,{f'{args.metric_name}/train':0},args.device)
    train_pbar=tqdm(train_loader,desc=f'epoch {epoch},train')
    model.train()
    loss_epoch=0
  
    for batch_id, (idxs, datas,labels) in enumerate(train_pbar):        
        """ forward and calculate loss  """
        #label to label_embeddings
        labels=labels.to(args.device)
        labels_em=embeddings[labels]
        #如果idxs!=None，可以输出错误样本的热力图等信息
        idxs=idxs.to(args.device)
        if epoch<args.debug_after['train']:
            idxs=None
        datas = datas.to(args.device)
        if args.use_amp:
            with amp_context(device_type='cuda'):
                out = model(datas,labels)
                preds_em=out[0]
                loss_batch,pred_labels=compute_loss_em(args.loss_name,out,embeddings,idxs,labels,datas)
        else:
            out = model(datas,labels)
            preds_em=out[0]
            loss_batch,pred_labels=compute_loss_em(args.loss_name,out,embeddings,idxs,labels,datas)   
        loss_epoch+=loss_batch
        metrics.add_batch(pred_labels,labels,preds_em,labels_em=datas)
        train_pbar.set_postfix(loss=loss_batch.item())
        """calculate gradient """
        if scaler is not None:
            scaler.scale(loss_batch).backward()
            scaler.step(optimizer)
            scaler.update() # next batch
        else:
            loss_batch.backward()
            optimizer.step()
        optimizer.zero_grad() 
        adjust_lr(wandb,schedule,epoch,batch_id,batchs)
    #logs
    retrun_metrics=metrics.compute_and_reset()
    loss_epoch=loss_epoch/batchs
    retrun_metrics['epoch']=epoch
    retrun_metrics['loss_epoch/train']=loss_epoch
    wandb.log(retrun_metrics)
    swanlab.log(retrun_metrics)

    

