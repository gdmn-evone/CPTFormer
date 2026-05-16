import torch
import numpy as np

from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from data import create_dataloader

def validate(model, opt):
    
    data_loader = create_dataloader(opt)
    with torch.no_grad():
        y_true, y_pred = [], []
        for data in tqdm(data_loader):
            input, _, input_phase, _, tf_label, _ = data
            input = input.cuda()
            input_phase = input_phase.cuda()
            tf_label = tf_label.cuda().long()
            tf_output, fea = model(input, input_phase)
            y_pred.extend(torch.argmax(tf_output, dim=1).flatten().tolist())
            y_true.extend(tf_label.flatten().tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)

    try:
        fpr, tpr, thresholds = metrics.roc_curve(y_true,y_pred, pos_label=1)
    except:
        return acc, ap, 0, None, r_acc, f_acc, y_true, y_pred

    if np.isnan(fpr[0]) or np.isnan(tpr[0]):
        auc, eer = 0, None
    else:
        auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return acc, ap, auc, eer, r_acc, f_acc, y_true, y_pred

