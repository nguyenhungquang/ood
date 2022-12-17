import torch
import torch.nn.functional as F
import numpy as np
from .base_detector import BaseDetector

from time import time
import logging

logger = logging.getLogger(__name__)

class pNMLDetector(BaseDetector):
    def __init__(self, net, *, require_grad=False, **kwargs) -> None:
        super().__init__("pNML", net, require_grad)
        self.x_t_x_inv = None
        self.pinv_rcond = 1e-15  # default
    
    def _forward_collect(self, data):
        out = self.net(data)
        features = self.net.get_features()
        probs = F.softmax(out, dim=1)
        return features, probs

    def setup(self, train_loader, **kwargs):
        with torch.no_grad():
            outputs = []
            device = "cuda:0"
            for data, target in train_loader:
                data = data.to(device)
                b_features, probs = self._forward_collect(data)
                outputs.append((probs, b_features))
            probs, b_features = zip(*outputs)
            probs = torch.vstack(probs)
            features = torch.vstack(b_features)
        
        t0 = time()
        x_t_x = torch.matmul(features.t(), features)
        _, s, _ = torch.svd(x_t_x, compute_uv=False)
        logger.info(f"Training set singular values largest: {s[:5]}")
        logger.info(f"Training set singular values smallest: {s[-5:]}")

        self.x_t_x = x_t_x
        self.x_t_x_inv = torch.linalg.pinv(
            self.x_t_x, hermitian=False, rcond=self.pinv_rcond
        )
        logger.info(f"Finish inv in {time() -t0 :.2f} sec")
    
    def score_batch(self, data):
        features, probs = self._forward_collect(data)
        features /= torch.linalg.norm(features, dim=-1, keepdim=True)
        logits_w_norm_features = self.net.fc(features)
        probs_normalized = torch.softmax(logits_w_norm_features, dim=-1)

        regrets = calc_regrets(self.x_t_x_inv, features, probs_normalized)
        preds = torch.argmax(probs, dim=1)
        return {
            "scores": regrets,
            "preds": preds
        }

    def __str__(self) -> str:
        return f"{self.name}"


def calc_regrets(x_t_x_inv: torch.Tensor, features: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    device = features.device
    x_t_x_inv = x_t_x_inv.type_as(features)
    x_t_x_inv.to(device)

    x_proj = torch.abs(
        torch.matmul(
            torch.matmul(features.unsqueeze(1), x_t_x_inv),
            features.unsqueeze(-1),
        ).squeeze(-1)
    )
    x_t_g = x_proj / (1 + x_proj)

    # Compute the normalization factor
    probs = probs.to(device)
    n_classes = probs.shape[-1]
    nf = torch.sum(probs / (probs + (1 - probs) * (probs ** x_t_g)), dim=-1)
    regrets = torch.log(nf) / torch.log(torch.tensor(n_classes))
    return regrets  - 1

def calc_regrets_custom(x_t_x_inv: torch.Tensor, features: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    device = features.device
    x_t_x_inv = x_t_x_inv.type_as(features)
    x_t_x_inv.to(device)

    x_proj = torch.abs(
        torch.matmul(
            torch.matmul(features.unsqueeze(1), x_t_x_inv),
            features.unsqueeze(-1),
        ).squeeze(-1)
    )
    x_t_g = x_proj / (1 + x_proj)

    # Compute the normalization factor
    probs = probs.to(device)
    n_classes = probs.shape[-1]
    nf = torch.sum(probs / (probs + (1 - probs) * (probs ** x_t_g)), dim=-1)
    regrets = torch.log(nf) / torch.log(torch.tensor(n_classes))
    
    # Higher is more OoD
    return regrets - 1
    # return x_t_g
    # return x_proj