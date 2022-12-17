import torch
import numpy as np
from .base_detector import BaseDetector

import logging

logger = logging.getLogger(__name__)

class GANDDetector(BaseDetector):
    def __init__(self, net, *, require_grad=False, **kwargs) -> None:
        super().__init__("ganD", net, require_grad)
    
    def _forward_collect(self, data):
        out = self.net(data)
        return torch.sigmoid(out)
    
    def score_batch(self, data):
        return self._forward_collect(data), torch.zeros((data.shape[0],), dtype=torch.int32)