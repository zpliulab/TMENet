"""Evaluation only: labels must never be supplied to the fitting functions."""
import numpy as np
from sklearn.metrics import average_precision_score,precision_recall_curve,roc_auc_score


def binary_metrics(truth,score):
    y=np.asarray(truth);s=np.asarray(score,dtype=float)
    if y.ndim!=1 or y.shape!=s.shape or not np.isin(y,[0,1]).all() or not np.isfinite(s).all():
        raise ValueError("aligned finite scores and binary labels are required")
    if np.unique(y).size!=2:raise ValueError("both positive and negative candidates are required")
    precision,recall,_=precision_recall_curve(y,s)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros_like(precision),where=(precision+recall)>0)
    return {"auroc":float(roc_auc_score(y,s)),"average_precision":float(average_precision_score(y,s)),
            "f1_max":float(f1.max()),"candidates":int(len(y)),"positives":int(y.sum())}
