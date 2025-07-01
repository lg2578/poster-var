import torch.nn.functional as F
import torch
from utils.visualization import tensors_show
def am_softmax_loss(logits, labels, s=10.0, m=0.5):
    one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()
    logits_m = logits - one_hot * m  # subtract margin from true class logit
    logits_m = logits_m * s  # optional scaling factor
    return F.cross_entropy(logits_m, labels,reduction='none')
def arcface_loss(cosine_sim, labels, s=30.0, m=0.20):
    one_hot = F.one_hot(labels, num_classes=cosine_sim.size(1)).float()
    theta = torch.acos(torch.clamp(cosine_sim, -1.0 + 1e-7, 1.0 - 1e-7))
    target_logits = torch.cos(theta + m)
    output = cosine_sim * (1 - one_hot) + target_logits * one_hot
    output = output * s
    return F.cross_entropy(output, labels,reduction='none')

def compute_loss_em(loss_name,out,embeddings,idxs,labels,datas,labels_em):
    '''
    idxs=None,则不做错误分析
    '''
    #计算输出向量和label向量拟合度
    if labels_em.size(1)==0:
        labels_em=embeddings[labels]

    preds_em=out[0] if type(out) is tuple else out
    scores = preds_em.float() @ embeddings.T
    #输出向量和label向量拟合度最高的两个值，第2名认为是难分样本
    topk_values, topk_idxs = torch.topk(scores, k=2, dim=1)
    pred_labels = topk_idxs[:, 0]
    candidate_labels=topk_idxs[:,1]
    candidate_labels_em=embeddings[candidate_labels]
    if loss_name=='cross_entropy':
        loss_batch=F.cross_entropy(preds_em,labels_em,reduction='none',label_smoothing=0.0)
    elif loss_name=='triplet_margin_loss':
        loss_batch=F.triplet_margin_loss(preds_em,labels_em,candidate_labels_em,margin=1.0)
    elif loss_name=='mse_loss':
        loss_batch=F.mse_loss(preds_em,labels_em,reduction='none')
    elif loss_name=='ce_kld_loss':
        mu=out[1]
        z_prior = F.softmax(torch.randn_like(mu))
        z =F.softmax(preds_em)
        loss_batch=F.cross_entropy(preds_em,labels_em,reduction='none')
        kl_loss=F.kl_div(z.log(),z_prior,reduction='none')
        kl_loss=torch.mean(kl_loss,dim=1)
        loss_batch+=kl_loss
    elif loss_name=='am_softmax_loss':
        loss_batch=am_softmax_loss(preds_em,labels)
    elif loss_name=='arcface_loss':

        loss_batch=arcface_loss(preds_em,labels)

    else:
        pass

    loss_batch_avg=torch.mean(loss_batch)
    if idxs!=None:
       #错误分类分析
        err_idxs=torch.nonzero(pred_labels!=labels)
        err_idxs=err_idxs.squeeze()
        if len(err_idxs)>0:
            print('error samples')
            print(idxs[err_idxs].detach().cpu().numpy(),labels[err_idxs].detach().cpu().numpy(),'->',pred_labels[err_idxs].detach().cpu().numpy())
        #损失值很大的样本
        ano_idxs=torch.where(loss_batch > 10 * loss_batch_avg)[0]
        if len(err_idxs)>0 and len(ano_idxs)>0:
            print('loss anomalous samples:')
            print(idxs[ano_idxs].detach().cpu().numpy(),\
                labels[ano_idxs].detach().cpu().numpy(),\
                loss_batch[ano_idxs].detach().cpu().numpy())
        tensors_show(datas[err_idxs])

    return loss_batch_avg,pred_labels
