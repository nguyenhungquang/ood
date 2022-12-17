import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy import misc
from .base_detector import BaseDetector

class GramDetector(BaseDetector):
    def __init__(self, name, net, require_grad=False) -> None:
        super().__init__("gram", net, require_grad)
        self.gram_mins = None
        self.gram_maxs = None
        self.ind_deviations = None
        self.dev_norm_sum = None
        self.ind_features = None
        self.ind_probs_normalized = None
    
    def _forward_collect(self, data):
        out, b_features = self.net.feature_list(data)
        b_gram_features = [gram_record(t) for t in b_features]
        probs = F.softmax(out, dim=1)
        return b_gram_features, probs

    def setup(self, train_loader):
        outputs = []
        for data, target in train_loader:
            b_gram_features, probs = self._forward_collect(data)
            gram_mins, gram_maxs = self._compute_minmaxs_train(
                b_gram_features, probs
            )
            outputs.append((probs, gram_mins, gram_maxs))
        probs, gram_mins, gram_maxs = zip(*outputs)
        probs = torch.vstack(probs)
        
        num_feature_layers = len(gram_mins[0][0])
        num_labels = probs.shape[-1]

        self.gram_mins =  {c: [None] * num_feature_layers for c in range(num_labels)}
        for c in range(num_labels):
            for l in range(num_feature_layers):
                # [batch, powers,values]
                mins_bathces = [
                    gram_mins_batch[c][l].unsqueeze(0)
                    for gram_mins_batch in gram_mins
                    if c in gram_mins_batch
                ]
                self.gram_mins[c][l] = torch.vstack(mins_bathces).min(dim=0).values

        self.gram_maxs = {c: [None] * num_feature_layers for c in range(num_labels)}
        for c in range(num_labels):
            for l in range(num_feature_layers):
                # [batch,powers,values]
                maxs_batches = [
                    gram_maxs_batch[c][l].unsqueeze(0)
                    for gram_maxs_batch in gram_maxs
                    if c in gram_maxs_batch
                ]
                self.gram_maxs[c][l] = torch.vstack(maxs_batches).max(dim=0).values
        return super().setup()
    
    def _compute_minmaxs_train(self, gram_feature_layers, probs):
        predictions = torch.argmax(probs, dim=1)

        predictions_unique = torch.unique(predictions)
        predictions_unique.sort()
        predictions_unique = predictions_unique.tolist()

        # Initialize outputs
        mins, maxs = {}, {}

        # Iterate on labels
        for c in predictions_unique:
            # Extract samples that are predicted as c
            class_idxs = torch.where(c == predictions)[0]
            gram_features_c = [
                gram_feature_in_layer_i[class_idxs]
                for gram_feature_in_layer_i in gram_feature_layers
            ]

            # Compute min and max of the gram features (per layer per power) shape=[layers,powers,features]
            mins_c = [layer.min(dim=0).values.cpu() for layer in gram_features_c]
            maxs_c = [layer.max(dim=0).values.cpu() for layer in gram_features_c]

            # Save
            mins[c] = mins_c
            maxs[c] = maxs_c

        return mins, maxs
    
    def score_batch(self, data):
        b_gram_features, probs = self._forward_collect(data)

        # Compute deviations of the test gram features from the min and max of the trainset
        deviations = self._compute_deviations(
            b_gram_features, probs
        )
        

    def _compute_deviations(self, b_gram_features, probs):
        # Initialize outputs
        deviations = []

        max_probs, predictions = probs.max(dim=1)  # [values, idxs]

        # Iterate on labels
        predictions_unique = torch.unique(predictions)
        predictions_unique.sort()
        predictions_unique = predictions_unique.tolist()

        for c in predictions_unique:
            # Initialize per class
            class_idxs = torch.where(c == predictions)[0]
            gram_features_per_class = [
                gram_feature_layer[class_idxs] for gram_feature_layer in b_gram_features
            ]
            max_probs_c = max_probs[predictions == c]

            deviations_c = get_deviations(
                gram_features_per_class, mins=self.gram_mins[c], maxs=self.gram_maxs[c]
            )
            deviations_c /= max_probs_c.to(deviations_c.device).unsqueeze(1)

            deviations.append(deviations_c)

        deviations = torch.cat(deviations, dim=0)

        return 

def get_deviations(features_per_layer_list, mins, maxs):
    '''
    Params:
        features_per_layer_list:
        mins:
        maxs:
    Return:
        deviations: [num_samples,num_layer]
    '''
    deviations = []
    for layer_num, features in enumerate(features_per_layer_list):
        layer_t = features  # [sample,power,value].
        mins_expand = mins[layer_num].unsqueeze(0)
        maxs_expand = maxs[layer_num].unsqueeze(0)

        # Divide each sample by the same min of the layer-power-feature
        layer_t = layer_t.to(mins_expand.device)
        devs_l = (
            torch.relu(mins_expand - layer_t) / torch.abs(mins_expand + 10 ** -6)
        ).sum(dim=(1, 2))
        devs_l += (
            torch.relu(layer_t - maxs_expand) / torch.abs(maxs_expand + 10 ** -6)
        ).sum(dim=(1, 2))
        deviations.append(devs_l.unsqueeze(1))
    deviations = torch.cat(deviations, dim=1)  # shape=[num_samples,num_layer]
    return deviations

# https://github.com/kobybibas/pnml_ood_detection/blob/39f1c5917724c0b0bab6ac7921dce3421b670236/src/model_arch_utils/gram_model_utils.py#L14
def G_p(ob, p):
    temp = ob.detach()
    temp = temp ** p
    temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
    temp = (torch.matmul(temp, temp.transpose(dim0=2, dim1=1))).sum(dim=2)
    temp = (temp.sign() * torch.abs(temp) ** (1 / p)).reshape(temp.shape[0], -1)
    return temp


def gram_record(t):
    feature = [G_p(t, p=p).unsqueeze(0) for p in range(1, 11)]
    feature = torch.cat(feature).transpose(0, 1)  # shape=[samples,powers,feature]
    return feature
