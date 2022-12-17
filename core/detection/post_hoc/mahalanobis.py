from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import numpy as np
from  .base_detector import BaseDetector


class MahalanobisDetector(BaseDetector):
    def __init__(self, net, *, noise_magnitude, layer_index=None, **kwargs) -> None:
        super().__init__('mahalanobis', net, True)
        self.net = net
        self.sample_mean = None
        self.precision = None
        self.magnitude = noise_magnitude
        self.layer_index = layer_index

    def setup(self, train_loader, num_classes):
        self.num_classes=num_classes
        self.net.eval()
        with torch.no_grad():
            feature_list = self.net.feature_list(torch.rand(2,3,32,32).cuda())[1]
            self.feature_size = np.array([out.size(1) for out in feature_list])

            print('get sample mean and covariance', len(self.feature_size))
            self.sample_mean, self.precision = sample_estimator(self.net, 
                                                    self.num_classes, 
                                                    self.feature_size, 
                                                    train_loader)
        if self.layer_index is None:
            self.layer_index = len(self.feature_size) - 1
    
    def score_batch(self, data):
        data = data.cuda()
        data = Variable(data, requires_grad = True)
        layer_index = self.layer_index

        out_features = self.net.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)
        
        # compute Mahalanobis score
        with torch.no_grad():
            gaussian_score = self._gaussian_score(out_features, layer_index)
            # Input_processing
            sample_pred = gaussian_score.argmax(1)

        batch_sample_mean = self.sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(self.precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
        
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
        
        tempInputs = torch.add(data.data, gradient, alpha=-self.magnitude)
        with torch.no_grad():
            noise_out_features = self.net.intermediate_forward(tempInputs, layer_index)
            noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
            noise_out_features = torch.mean(noise_out_features, 2)
            noise_gaussian_score = self._gaussian_score(noise_out_features, layer_index)
            noise_gaussian_score = torch.max(noise_gaussian_score, dim=1).values
        return {
            "scores": -noise_gaussian_score,
            "preds": sample_pred
        }
    
    def _gaussian_score(self, out_features, layer_index):
        gaussian_score = []
        for i in range(self.num_classes):
            zero_f = out_features.data - self.sample_mean[layer_index][i]
            term_gau = -0.5*torch.mm(torch.mm(zero_f, self.precision[layer_index]), zero_f.t()).diag()
            gaussian_score.append(term_gau.view(-1,1))
        return torch.cat(gaussian_score, 1)
    
    def __str__(self) -> str:
        return f"{self.name}_noise={self.magnitude}_layerindex={self.layer_index}"


def sample_estimator(model, num_classes, feature_list, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance
    
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    list_features = [[None]*num_classes for _ in range(num_output)]
    
    for data, target in train_loader:
        total += data.size(0)
        data = data.cuda()
        output, out_features = model.feature_list(data)
        
        # get hidden features
        for i in range(num_output):
            out_features[i] = torch.mean(out_features[i].data, list(range(2, out_features[i].ndim)))
            
        # compute the accuracy
        pred = output.data.argmax(1)
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()

        for label in range(num_classes):
            inds = target == label
            if inds.sum() == 0: continue
            for out_count, out in enumerate(out_features):
                tmp = out[inds]
                tmp = tmp.view(tmp.size(0), -1)
                if list_features[out_count][label] is not None:
                    list_features[out_count][label].append(tmp)
                else:
                    list_features[out_count][label] = [tmp]
    
    sample_class_mean = []
    precision = []
    for k in range(num_output):
        temp_list = torch.Tensor(num_classes, feature_list[k]).cuda()
        for j in range(num_classes):
            if list_features[k][j] is not None:
                list_features[k][j] = torch.cat(list_features[k][j], dim=0)
                temp_list[j] = torch.mean(list_features[k][j], 0)
                list_features[k][j] -= temp_list[j]
        sample_class_mean.append(temp_list)
        X = torch.cat(list_features[k], dim=0)

        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)

    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))
    return sample_class_mean, precision
