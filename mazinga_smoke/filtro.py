import torch
import torch.nn as nn

class Filtro(nn.Module):

    def __init__(self, c, k=5, init_gain=0.5):
        
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=k, stride=1, padding=k//2)
        self.gain = nn.Parameter(torch.ones(1, c, 1, 1) * init_gain)
        self.trace = False
        self.diagnostic_data = {}

    def forward(self, x):

        lp = self.pool(x)
        hp = x - lp
        y  = x + self.gain * hp

        return y
