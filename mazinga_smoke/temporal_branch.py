import torch
import torch.nn as nn
import torch.nn.functional as F
from .dws import DepthwiseSeparableBlock



class TemporalBranch(nn.Module):
    """
    - estrazione feature (mean/std/dx) da int
    - fusione con deep
    - proiezione a d_model
    - TCN short + TCN long in parallelo
    - somma: x_mix = x_short + x_long
    - attention sui frame + pooling pesato
    """
    def __init__(self, d_model=128, c_int=40, n_layers=None, use_dilation=None, tau=None):
        super().__init__()
        self.d_model = d_model
        
        # 1. ADATTATORE feature intermedie
        self.int_adapter = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(c_int, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 14x14 -> 7x7
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # 128 (deep) + 16 (mean) + 16 (std) + 16 (dx) = 176
        dim_in = d_model + 16 + 16 + 16

        # 2. PROIEZIONE
        self.proj_linear = nn.Linear(dim_in, d_model, bias=False)  # [B, T, D]
        self.proj_bn     = nn.BatchNorm1d(d_model)                 # [B, D, T]
        self.proj_act    = nn.GELU()

        # 3. TCN SHORT 
        self.tcn_short = nn.Sequential(
            DepthwiseSeparableBlock(d_model, kernel_size=3, dilation=1),
            DepthwiseSeparableBlock(d_model, kernel_size=3, dilation=2),
        )

        # 4. TCN LONG 
        self.tcn_long = nn.Sequential(
            DepthwiseSeparableBlock(d_model, kernel_size=3, dilation=1),
            DepthwiseSeparableBlock(d_model, kernel_size=3, dilation=2),
            DepthwiseSeparableBlock(d_model, kernel_size=3, dilation=4),
            DepthwiseSeparableBlock(d_model, kernel_size=3, dilation=8),
        )

        # 5. HEAD DI ATTENZIONE E NORMALIZZAZIONE
        self.w_head   = nn.Linear(d_model, 1)
        self.norm_out = nn.LayerNorm(d_model)

        self.frame_importance = None

    def forward(self, f_deep, f_int):
        """
        f_deep : [B, T, D, H, W]
        f_int: [B, T, Ce, He, We]
        Ritorna:
          f_temp     : [B, D]
          w          : [B, T]
          x_seq_out  : [B, T, D] 
        """
        B, T, D, H, W = f_deep.shape

        flat_int = f_int.view(B * T, -1, f_int.size(3), f_int.size(4))
        feat_int_map = self.int_adapter(flat_int)            # [B*T, 16, 7, 7]
        feat_int_vec = feat_int_map.view(B, T, 16, -1)         # [B, T, 16, 49]

        x_mean = feat_int_vec.mean(dim=-1)                       # [B, T, 16]
        x_std  = feat_int_vec.std(dim=-1)                        # [B, T, 16]
        dx = torch.zeros_like(x_mean)
        dx[:, 1:] = x_mean[:, 1:] - x_mean[:, :-1]                 # [B, T, 16]

        # DEEP FEATURES (global pooling + max) 
        x_deep = f_deep.mean(dim=(3, 4)) + f_deep.amax(dim=(3, 4)) # [B, T, 128]

        # FUSIONE canali
        x_combined = torch.cat([x_deep, x_mean, x_std, dx], dim=-1)  # [B, T, 176]

        # PROIEZIONE A d_model
        x_seq = self.proj_linear(x_combined)   # [B, T, D]
        x_seq = x_seq.transpose(1, 2)          # [B, D, T]
        x_seq = self.proj_bn(x_seq)            
        x_seq = self.proj_act(x_seq)           # [B, D, T]

        # DUE RAMI TCN IN PARALLELO
        x_short = self.tcn_short(x_seq)        # [B, D, T] 
        x_long  = self.tcn_long(x_seq)         # [B, D, T] 

        # SOMMA
        x_mix = x_short + x_long               # [B, D, T]
        x_mix = x_mix.transpose(1, 2)          # [B, T, D]

        # ATTENZIONE SUI FRAME 
        logits_w = self.w_head(x_mix).squeeze(-1)  # [B, T]
        w = torch.softmax(logits_w, dim=-1)        # [B, T]

        # POOLING PESATO 
        x_seq_out = self.norm_out(x_mix)           # [B, T, D]
        f_temp = (x_seq_out * w.unsqueeze(-1)).sum(dim=1)  # [B, D]

        # Grad-CAM
        self.frame_importance = w.detach().cpu()

        return f_temp, w, x_seq_out 

