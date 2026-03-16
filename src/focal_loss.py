import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLossBalanced(nn.Module):
    def __init__(self, alpha_pos=0.6, alpha_neg=0.4, gamma=2.0, reduction='none'):

        super().__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()

        # prob
        p = torch.sigmoid(logits)

        # BCE per sample
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # alpha mixata
        alpha_t = targets * self.alpha_pos + (1.0 - targets) * self.alpha_neg

        # pt per soft target: p se target=1, (1-p) se target=0
        pt = targets * p + (1.0 - targets) * (1.0 - p)

        loss = alpha_t * (1.0 - pt).pow(self.gamma) * bce
        return loss  