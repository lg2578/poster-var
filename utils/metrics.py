import numpy as np
from sklearn.metrics import roc_auc_score,confusion_matrix,f1_score
from torchmetrics import Accuracy,ConfusionMatrix,F1Score,Precision,Recall,Specificity,AUROC
from torchmetrics.regression import MeanAbsoluteError,MeanSquaredError
from utils.visualization import plot_confusion_matrix
import wandb

class Metrics(object):
    def __init__(self,num_classes,return_metrics,device):
        '''
        return_metrics:{'acc':0,'f1':0}
        '''
        super(Metrics,self).__init__()
        self.metrics_label={}
        self.metrics_em={}
        self.metrics_em_label={}
        self.return_metrics=return_metrics
        for k in self.return_metrics:
            if 'acc' in k:
                self.metrics_label[k]=Accuracy(task='multiclass',num_classes=num_classes).to(device)
            if 'confusion_matrix' in k:
                self.metrics_label[k]=ConfusionMatrix('multiclass',num_classes=num_classes).to(device)
            if 'f1' in k:
                self.metrics_label[k]=F1Score('multiclass',num_classes=num_classes).to(device)
            if 'precision' in k:
                self.metrics_label[k]=Precision('multiclass',num_classes=num_classes).to(device)
            if 'recall' in k:    
                self.metrics_label[k]=Recall('multiclass',num_classes=num_classes).to(device)
            if 'specificity' in k:
                self.metrics_label[k]=Specificity('multiclass',num_classes=num_classes).to(device)
            if 'auroc' in k:
                self.metrics_em_label[k]=AUROC('multiclass',num_classes=num_classes).to(device)
            if 'mae' in k:
                self.metrics_em[k]=MeanAbsoluteError().to(device)
            if 'mse' in k:
                self.metrics_em[k]=MeanSquaredError().to(device)
            if 'rmse' in k:
                self.metrics_em[k]=MeanSquaredError(squared=False).to(device)


    def add_batch(self,preds_labels,labels,preds_em,labels_em):
        for k in self.metrics_label:
            self.metrics_label[k](preds_labels,labels)
        for k in self.metrics_em:
            self.metrics_em[k](preds_em,labels_em)
        for k in self.metrics_em_label:
            self.metrics_em_label[k](preds_em,labels)
    def compute_and_reset(self,):
        for k in self.metrics_label:
            self.return_metrics[k]=self.metrics_label[k].compute()
            self.metrics_label[k].reset()
            if 'confusion_matrix' in k:
                cm=self.return_metrics[k]
                fig=plot_confusion_matrix(cm.cpu().numpy())
                wandb_image = wandb.Image(fig)
                self.return_metrics[k]=wandb_image
        for k in self.metrics_em:
            self.return_metrics[k]=self.metrics_em[k].compute()
            self.metrics_em[k].reset()
        for k in self.metrics_em_label:
            self.return_metrics[k]=self.metrics_em_label[k].compute()
            self.metrics_em_label[k].reset()
        return self.return_metrics

def comp_bc_scores(y_true, y_pred, min_tpr=0.80):
    v_gt = abs(y_true-1)
    v_pred = np.array([1.0 - x for x in y_pred])
    max_fpr = abs(1-min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    # 将预测值二值化
    threshold = 0.5  # 常用阈值，可根据需要调整
    y_pred_binary = [1 if p >= threshold else 0 for p in y_pred]

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    # 计算敏感性（召回率/True Positive Rate）
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    # 计算特异性（True Negative Rate）
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # 计算准确率
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # 计算精确率（Precision）
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # 计算 AUC
    auc = roc_auc_score(y_true, y_pred)

    # 计算 F1 Score
    f1 = f1_score(y_true, y_pred_binary)
    # print(f'metrics:{tn=},{fp=},{fn=},{tp=},{sensitivity=},{specificity=},{accuracy=},{auc=},{f1=}')
    metrics = {
        'pAUC':round(partial_auc,5),   
        'tn': round(tn,5),
        'fp': round(fp,5),
        'fn': round(fn,5),
        'tp': round(tp,5),
        'sensitivity': round(sensitivity,5),
        'specificity': round(specificity,5),
        'accuracy': round(accuracy,5),
        'precision':round(precision,5),
        'auc': round(auc,5),
        'f1': round(f1,5)
    }
    return metrics
