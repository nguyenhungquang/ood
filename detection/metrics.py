import numpy as np

def tpr95(in_score, out_score, start, end, step=100000):
    total = 0
    fpr = 0
    step_size = (end - start) / step
    for delta in np.arange(start, end, step_size):
        tpr = np.sum(in_score >= delta) / float(len(in_score))
        err = np.sum(out_score > delta) / float(len(out_score))
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += err
            total += 1
    fpr = fpr / total
    return fpr

def auroc(in_score, out_score, start, end, step=100000):
    step_size = (end - start) / step
    auroc = 0
    temp_fpr = 1
    for delta in np.arange(start, end, step_size):
        tpr = np.sum(in_score >= delta) / float(len(in_score))
        fpr = np.sum(out_score >= delta) / float(len(out_score))
        auroc += (temp_fpr - fpr) * tpr
        temp_fpr = fpr
    auroc += fpr * tpr
    return auroc

def detection_err(in_score, out_score, start, end, step=100000):
    step_size = (end - start) / step
    err = 1
    for delta in np.arange(start, end, step_size):
        tpr = np.sum(in_score < delta) / float(len(in_score))
        err2 = np.sum(out_score > delta) / float(len(out_score))
        err = np.minimum(err, (tpr + err2) / 2)
    return err
