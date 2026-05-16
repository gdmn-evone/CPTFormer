import torch
import numpy as np

from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from data import create_dataloader
from data.datasets_phase import *
import torch.distributed as dist

def validate(model, opt, world_size=1, rank=0):
    
    data_loader = create_dataloader(opt, world_size, rank)
    with torch.no_grad():
        local_y_pred, local_y_true = [], []
        for data in tqdm(data_loader, disable=(rank != 0), desc="Validating"):
            input, _, input_phase, _, tf_label, _ = data
            input = input.cuda(non_blocking=True)
            input_phase = input_phase.cuda(non_blocking=True)
            tf_label = tf_label.cuda(non_blocking=True)
            tf_output, _ = model(input, input_phase)
            tf_output = model(input)
            local_y_pred.extend(torch.argmax(tf_output, dim=1).flatten().cpu().tolist())
            local_y_true.extend(tf_label.flatten().cpu().tolist())
    
    all_proc_y_true = [None] * world_size
    all_proc_y_pred = [None] * world_size

    dist.all_gather_object(all_proc_y_true, local_y_true)
    dist.all_gather_object(all_proc_y_pred, local_y_pred)
    if rank == 0:
        
        y_true = [item for sublist in all_proc_y_true for item in sublist]
        y_pred = [item for sublist in all_proc_y_pred for item in sublist]
        
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        num_samples = len(data_loader.dataset)
        y_true = y_true[:num_samples]
        y_pred = y_pred[:num_samples]

        r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
        f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
        acc = accuracy_score(y_true, y_pred > 0.5)
        ap = average_precision_score(y_true, y_pred)

        try:
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        except ValueError:
            return acc, ap, 0, None, r_acc, f_acc, y_true, y_pred

        if np.isnan(fpr).any() or np.isnan(tpr).any():
            auc, eer = 0, None
        else:
            auc = metrics.auc(fpr, tpr)
            fnr = 1 - tpr
            eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
            eer = fpr[eer_idx]
            
        return acc, ap, auc, eer, r_acc, f_acc, y_true, y_pred
    else:
        return (None,) * 8

