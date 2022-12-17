import torch
import torch.nn.functional as F
import numpy as np
from  .base_detector import BaseDetector

to_np = lambda x: x.data.cpu().numpy()
concat = lambda x: np.concatenate(x, axis=0)

energy_score = lambda smax, output, T: -T*torch.logsumexp(output / T, dim=1)
msp_score = lambda smax, output=None, T=None: -torch.max(smax, dim=1).values #TODO: why minus?
xent_score = lambda smax, output, T: output.mean(1) - torch.logsumexp(output, dim=1)

class MSPDetector(BaseDetector):
    def __init__(self, net, use_xent=False, T=None, *args, **kwargs) -> None:
        super().__init__('msp', net)
        self.use_xent = use_xent
        self.T = T
        if self.use_xent:
            self.score_fn = lambda smax, output: xent_score(smax, output, self.T)
        else:
            self.score_fn = lambda smax, output=None: -torch.max(smax, dim=1).values #TODO: why minus?

    def score(self, loader, in_dist=False):
        return self._get_ood_scores(loader, in_dist=in_dist)
    
    def score_batch(self, data):
        output = self.net(data)
        smax = F.softmax(output, dim=1)
        preds = torch.argmax(smax, dim=1)
        return {
            "scores": self.score_fn(smax, output), 
            "preds": preds
        }

    def __str__(self) -> str:
        return f"{self.name}_T={self.T}_use_xent={self.use_xent}"

class EnergyBasedDetector(MSPDetector):
    def __init__(self, net, T, *args, **kwargs) -> None:
        super().__init__(net)
        self.name = 'energy'
        self.T = T
        self.score_fn = lambda smax, output: energy_score(smax, output, self.T)

    def __str__(self) -> str:
        return f"{self.name}_T={self.T}"
