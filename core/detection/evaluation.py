from pathlib import Path
import torch
import torch.functional as F
from torch.utils.data import Subset
import numpy as np
import sklearn.metrics as sk
from torchmetrics import Accuracy
import pandas as pd
from ood_metrics import calc_metrics
from tqdm import tqdm

RECALL_LEVEL_DEFAULT = 0.95
def get_measures(_pos, _neg, recall_level=RECALL_LEVEL_DEFAULT):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    metric_dict = calc_metrics(examples, labels)

    # auroc = sk.roc_auc_score(labels, examples)
    # aupr = sk.average_precision_score(labels, examples)
    # fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    auroc = metric_dict['auroc']
    aupr = metric_dict['aupr_in']
    fpr = metric_dict['fpr_at_95_tpr']

    return auroc, aupr, fpr

def score_loop(detector, loader, in_dist=False):
    # _right_score = []
    # _wrong_score = []
    # metric = MetricCollection([Accuracy(), AUROC, AUC])
    accuracy_metric = Accuracy().cuda()
    scores = None
    def loop():
        _score = []
        for data, target in tqdm(loader):
            data, target = data.cuda(), target.cuda()
            result_dict = detector.score_batch(data)
            _score.append(result_dict['scores'])

            if in_dist and 'preds' in result_dict:
                accuracy_metric.update(result_dict['preds'], target)
        return torch.cat(_score, dim=0).cpu()

    if detector.require_grad:
        scores = loop()
    else:
        with torch.no_grad():
            scores = loop()
    
    if in_dist:
    #     return (concat(_score).copy(),
    #             concat(_right_score).copy(), 
    #             concat(_wrong_score).copy())
        return scores, {'accuracy': accuracy_metric.compute()}
        # return scores, {'accuracy': 0}

    # else:
    #     return concat(_score).copy()
    return scores
    

def _random_subset_loader(dataset, num_examples, bs, num_workers=2):
    indices = torch.randperm(len(dataset))[:num_examples]
    # ood_data_indices = np.arange(ood_num_examples)
    sub_data = Subset(dataset, indices)
    return torch.utils.data.DataLoader(sub_data, 
                                        batch_size=bs, 
                                        shuffle=False,
                                        num_workers=num_workers, 
                                        pin_memory=True)

def get_results(detector, get_measures, ood_data, num_examples, batch_size, num_runs=5):
    measures = []
    for _ in range(num_runs):
        ood_loader = _random_subset_loader(ood_data, num_examples, batch_size)
        out_score = score_loop(detector, ood_loader, in_dist=False)
        out_score = out_score.numpy()
        measures.append(get_measures(out_score))
    measures = zip(*measures)
    return measures

class Evaluator():
    def __init__(self, detector, ood_num_examples, test_bs, num_to_avg: int) -> None:
        self.detector = detector
        self.in_score = None
        # self.get_measures = get_measures
        self.ood_num_examples = ood_num_examples
        self.test_bs = test_bs
        self.num_to_avg = num_to_avg
        # self.in_score = in_score.copy() #TODO: Store reference?
        self.df = pd.DataFrame(columns=("ood_data", "num_runs", 
                                        "auroc", "aupr", "fpr", 
                                        "auroc_std", "aupr_std", 
                                        "fpr_std"))
    
    def compute_in_score(self, in_loader):
        in_score, metrics = score_loop(self.detector, 
                                       in_loader, 
                                       in_dist=True)
        self.in_score = in_score.numpy()
        print('Error Rate {:.2f}'.format(100 * (1 - metrics['accuracy'])))

    
    def eval_ood(self, ood_name, ood_data, verbose=False):
        assert self.in_score is not None
        aurocs, auprs, fprs = get_results(self.detector, 
                            lambda out_scores: get_measures(-self.in_score, -out_scores),
                            ood_data,
                            self.ood_num_examples,
                            self.test_bs,
                            self.num_to_avg)
        auroc = 100*np.mean(aurocs); aupr = 100*np.mean(auprs); fpr = 100*np.mean(fprs)
        record = [ood_name, self.num_to_avg, auroc, aupr, fpr]
        record.extend([100*np.std(aurocs),100*np.std(auprs),100*np.std(fprs)]) 
        self.df.loc[len(self.df)] = record
        if verbose:
            print(self.df.loc[-1])

    def reset(self):
        self.df = pd.DataFrame(columns=("ood_data", "num_runs", 
                                        "auroc", "aupr", "fpr", 
                                        "auroc_std", "aupr_std", 
                                        "fpr_std"))

    def save(self, dir_path):
        dir_path = Path(dir_path)
        self.df.to_csv(dir_path / f"{self.detector}.csv", index=False)