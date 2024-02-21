import torch
from typing import List
from torch import nn, Tensor

class MRLLoss(nn.Module):
    def __init__(self, cm: List[int]):
        super(MRLLoss, self).__init__()

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, nested_preds: Tensor, ground_truth: Tensor, cm: Tensor = None):
        if cm is None:
            cm = torch.ones_like(ground_truth[0])

        mrl_losses = torch.stack([self.ce_loss(pred, ground_truth) for pred in nested_preds])
        loss = mrl_losses * cm
        
        return loss.sum()