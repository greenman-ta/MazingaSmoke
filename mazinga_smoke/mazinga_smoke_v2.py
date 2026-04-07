import torch
import torch.nn as nn

from .backbone_ghostnet import GhostV2Backbone
from .static_branch import StaticBranch  
from .temporal_branch import TemporalBranch 
from .ca_fusion_multi_mod import CAFusion 
from .cepool import CEPool


class MazingaSmokeClassifier(nn.Module):

    def __init__(self, T,
                 backbone_pretrained: bool = True,
                  d_mod: int = 128, 
                  k_max: int = 4):
        super().__init__()
        self.T = T

        #init moduli
        self.backbone = GhostV2Backbone(pretrained=backbone_pretrained, T=T)
        self.static_branch = StaticBranch(C_in=128, T=T)
        self.temporal_branch = TemporalBranch(
            d_model=d_mod,
            c_int=self.backbone.C_int
        )
        self.fusion_module = CAFusion()

        self.static_in_proj = nn.Sequential(
            nn.Conv2d(d_mod + d_mod, d_mod, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_mod),
            nn.ReLU(inplace=True),

        )
        
        self.static_gate = nn.Linear(d_mod, 1) 
        self.cepool = CEPool(c_in=d_mod, c_out=d_mod, stride=4)  
        self.delta_proj = nn.Conv2d(self.backbone.C_int, d_mod, kernel_size=1, bias=False)

        #teste

        self.aux_head = nn.Sequential(
            nn.LayerNorm(d_mod),
            nn.Linear(d_mod, d_mod // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_mod // 2, 1),
        )

        self.mlp = nn.Sequential(
            nn.LayerNorm(d_mod),             
            nn.Linear(d_mod, d_mod // 2),
            nn.SiLU(),
            nn.Dropout(0.1),                 
            nn.Linear(d_mod // 2, d_mod // 4),
            nn.SiLU(),
            nn.Dropout(0.1),                 
            nn.Linear(d_mod // 4, 1)
        )

        self.static_head = nn.Sequential(
            nn.LayerNorm(d_mod),
            nn.Linear(d_mod, d_mod // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_mod // 2, 1),
        )
        self.tau_static = nn.Parameter(torch.tensor(0.7)) 

        self.last_vis = {}


    def _build_static_in(self, f_k, f_int_k, topK):

        B, T, Cd, Hd, Wd = f_k.shape
        _, _, Ce, He, We = f_int_k.shape
        K = topK.shape[1]

        topK_prev = (topK - 1).clamp(min=0)
        idx_prev  = topK_prev[..., None, None, None].expand(B, K, Ce, He, We)
        f_int_prev = f_int_k.gather(1, idx_prev) # [B,K,Ce,He,We]

        # flatten 2D
        BKT = B * K
        deep_2d = f_k.reshape(BKT, Cd, Hd, Wd)
        int_2d  = f_int_k.reshape(BKT, Ce, He, We)
        prev_2d = f_int_prev.reshape(BKT, Ce, He, We)

        delta_raw = int_2d - prev_2d
        delta_128 = self.delta_proj(delta_raw)   # [B*K,128,He,We]
        delta_ds  = self.cepool(delta_128)       # [B*K,128,Hd,Wd]

        cat = torch.cat([deep_2d, delta_ds], dim=1)  # [B*K,Cd+128,Hd,Wd]
        static_2d = self.static_in_proj(cat)         # [B*K,128,Hd,Wd]
        static_in = static_2d.view(B, K, d_mod, Hd, Wd)
        return static_in


    def forward(self, x, d_mod=128, ft=False):
        
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)

        # --- BACKBONE GHOSTNETV2 ---
        f_deep, f_int = self.backbone(x)
        # f_int: [B,T,40,H_int,W_int] strato intermedio
        # f_deep : [B,T,128,H_deep,W_deep] ultimo strato

        # --- TEMPORAL BRANCH ---
        f_temp, w, x_seq = self.temporal_branch(f_deep, f_int)  #3 uscite, 1 per testa, 2 pesi attn, 3 feature seq temporali

        # adattamento per V2: prendo tutti i frame come topK per riadattare static branch e fusione
        K = self.T
        topk_idx = torch.arange(T, device=f_deep.device).unsqueeze(0).expand(B, -1)
        w_topk = w
        
        # costruisco input static branch
        static_in = self._build_static_in(f_deep, f_int, topk_idx)

        # --- STATIC BRANCH ---
        f_static = self.static_branch(static_in)   # [B,K,128,Hd,Wd]

        # --- FUSIONE ---
        f_k, attn_mapK, f_fused = self.fusion_module(f_static, x_seq, topk_idx=topk_idx, w_topk=w_topk) 

        # --- TESTE CLASSIFICAZIONE ---
        
        # principale
        logit = self.mlp(f_fused).squeeze(-1)         # [B]

        # temporale
        aux_logit = self.aux_head(f_temp).squeeze(-1) # [B]

        # statico
        B, K, C, H, W = f_static.shape

        # pooling ibrido sulle feature (mean+max)
        stat_k = 0.5 * f_static.mean(dim=(3,4)) + 0.5 * f_static.amax(dim=(3,4))  # [B,K,128]

        # logits per frame
        frame_logit = self.static_head(stat_k.view(B*K, C)).view(B, K)  # [B,K]

        # pesi su K 
        tau = torch.clamp(self.tau_static, 0.2, 2.0)

        scores = self.static_gate(stat_k).squeeze(-1)   # [B,K]
        w_s = torch.softmax(scores / tau, dim=1)
        static_logit = (frame_logit * w_s).sum(dim=1)

        # --- debug gradcam---
        try:
            self.last_vis = {
                "topk_idx": torch.arange(T, device=f_static.device).detach().cpu(),
                "topk_order": torch.arange(T, device=f_static.device).detach().cpu(),
                "post_attn_maps_k": attn_mapK[0].detach().cpu(),
                "w_topk": None,
            }
        except Exception:
            self.last_vis = {}

        # --- OUTPUT ---
        if ft:
            return logit, aux_logit, static_logit, f_fused
        elif self.training:
            return logit, aux_logit, static_logit
        else:
            return logit
    

    