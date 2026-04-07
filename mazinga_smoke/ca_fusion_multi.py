import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CAFusion(nn.Module):
    def __init__(self, dim=128, num_heads=4, dp=0.05, H=7, W=7):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.H = H
        self.W = W

        # proiezioni lineari per query, key, value
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj   = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

        self.pos2d = nn.Parameter(torch.zeros(1, 1, dim, H, W))
        nn.init.trunc_normal_(self.pos2d, std=0.02)
        
        # layer finale per combinare output 
        self.out_proj = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp)

        # norme
        self.norm1   = nn.LayerNorm(dim)
        self.norm2   = nn.LayerNorm(dim)

        # feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dp),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dp)
        )

        # bilanciamento
        self.tau_spat = nn.Parameter(torch.tensor(0.0))   # temperatura spaziale learnable
        self.tau_gate = nn.Parameter(torch.tensor(1.0))   # temperatura gate K learnable
        self.gate = nn.Linear(self.dim, 1)   # score s_k [B,K,1]

        # proiezione per fusione concatenata [m_flat, y_norm] -> dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
        # logging
        self.last_w = None            # [B,K] pesi temporali effettivi
        self.last_topk_idx = None     # [B,K] se usi topk

        # cam 
        self.capture_cam = False
        self.last_feat = None 

        self.norm_y  = nn.LayerNorm(dim)


    def forward(self, s_in, m_in, topk_idx=None, w_topk=None):

        B, K, C, H, W = s_in.shape
        N = H * W

        # positional 2D
        pos2d = self.pos2d
        s_pos = s_in + pos2d                      # [B,K,C,H,W]

        # flatten spaziale
        s_flat = s_pos.view(B*K, C, H, W)         # [B*K,C,H,W]
        tok0 = s_flat.flatten(2).transpose(1, 2)  # [B*K,N,C]

        # flatten temporale per segmento
        m_q = m_in.view(B*K, self.dim)         # [B*K,dim]

        tok_C = tok0.transpose(1, 2).contiguous().view(B, K, C, H, W)   # [B,K,C,H,W]
        if self.capture_cam:
            self.last_feat = tok_C
            self.last_feat.retain_grad()  # per Grad-CAM

        tok = tok_C.view(B*K, C, H, W).flatten(2).transpose(1, 2)       # [B*K,N,C]

        q_lin = self.query_proj(m_q)   # [B*K,dim]
        k_lin = self.key_proj(tok)     # [B*K,N,dim]
        v_lin = self.value_proj(tok)   # [B*K,N,dim]

        # proiezioni per multihead attention
        q = q_lin.view(B*K, self.num_heads, 1, self.dim_head)                 # [B*K,h,1,d_h]
        k = k_lin.view(B*K, N, self.num_heads, self.dim_head).transpose(1,2)  # [B*K,h,N,d_h]
        v = v_lin.view(B*K, N, self.num_heads, self.dim_head).transpose(1,2)  # [B*K,h,N,d_h]
        
        # 2. cross-attention spaziale
        att = q @ k.transpose(-2, -1) / (self.dim_head ** 0.5)  # [B*K,h,1,N]
        tau_spat = 0.7 + 0.8 * torch.sigmoid(self.tau_spat)
        attn = torch.softmax(att / tau_spat, dim=-1)            # [B*K,h,1,N]

        y = (attn @ v).transpose(1, 2).contiguous().view(B*K, self.dim)  # [B*K,dim]
        y = self.dp(self.out_proj(y)) 
        y_norm = self.norm_y(y)                                # [B*K,dim]

        # FUSIONE: CONCAT [temporale, fusione]
        fused_cat = torch.cat([m_q, y_norm], dim=-1)      # [B*K, 2*dim]
        x = self.fusion_proj(fused_cat)                   # [B*K, dim]

        # residual + FFN
        x = x + self.ffn(self.norm2(x))     # [B*K,dim]

        z_k = x.view(B, K, self.dim)        # [B,K,dim]

        # mappa di attenzione spaziale per K
        att_mapK = attn.mean(dim=1).squeeze(1).view(B, K, H, W)          # [B,K,H,W]
        # normalizzo per CAM
        att_mapK = att_mapK / (att_mapK.amax((2,3), keepdim=True) + 1e-8)

        # gating sui K 
        scores = self.gate(z_k).squeeze(-1)                             # [B,K]
        tau_gate = torch.clamp(self.tau_gate, 0.8, 1.8)
        w_gate  = torch.softmax(scores / tau_gate, dim=1)               # [B,K]

        #  mix di esperti
        w_eff = w_topk * w_gate                                     # [B,K]
        w_eff = w_eff / w_eff.sum(dim=1, keepdim=True).clamp_min(1e-8)

        # logging
        self.last_w = w_eff.detach()
        self.last_topk_idx = topk_idx.detach().cpu()

        # pooling MIL sulle feature di segmento
        z_fused = (z_k * w_eff[..., None]).sum(dim=1)  # [B,dim]
        
        return z_k, att_mapK, z_fused
    
    def clear_cam_cache(self):
        self.last_feat = None