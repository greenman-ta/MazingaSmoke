import torch
import torch.nn as nn
import torch.nn.functional as F
from .filtro import Filtro
from .ceconv import CEConv

# Le dimensioni latenti di questo modulo rimangono hardcoddate per rendere compatibili i pesi rilasciati nella repository

class StaticBranch(nn.Module):
    def __init__(self, C_in=128, T: int = 4, drop_p: float = 0.10):
        super().__init__()
        self.T = T

        self.filtro = Filtro(c=C_in, k=5, init_gain=0.5)
        self.mc = CEConv(C_in, drop_p=drop_p)

        self.spatial_dropout = nn.Dropout2d(p=drop_p)

        self.fc_out   = nn.Linear(C_in, 128)
        self.norm_out = nn.LayerNorm(128)

        self.capture_cam = False
        self.last_feat   = None

    def reshape(self, z2d):
        # z2d: [B*K, C_in, H, W] -> [B*K, 128, H, W]
        z = z2d.permute(0, 2, 3, 1)     # [N,H,W,C]
        z = self.fc_out(z)             # [N,H,W,128]
        z = self.norm_out(z)
        z = z.permute(0, 3, 1, 2)      # [N,128,H,W]
        return z

    def forward(self, x):

        B, K, C, H, W = x.shape
        x2d = x.view(B * K, C, H, W)   # [B*K,C,H,W]

        y1 = self.filtro(x2d)        # [B*K,C,H,W]
        y2 = self.mc(y1)               # [B*K,C,H,W]
        y2_pre_drop = y2

        y2_drop = self.spatial_dropout(y2_pre_drop)     # [B*K, C, H, W]
        
        # Grad-CAM hook 
        if self.capture_cam and torch.is_grad_enabled():
            self.last_feat = y2_drop
            self.last_feat.retain_grad()

        # reshape
        x128 = self.reshape(y2_drop)                      # [B*K, 128, H, W]
        x_out   = x128.view(B, K, 128, H, W)

        return x_out
