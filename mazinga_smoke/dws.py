import torch
import torch.nn as nn

class DepthwiseSeparableBlock(nn.Module):
    """
    Per TCN: Depthwise + Pointwise.
    """
    def __init__(self, dim, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        
        self.net = nn.Sequential(
            # Depthwise
            nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation, groups=dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            # Pointwise
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.net(x)