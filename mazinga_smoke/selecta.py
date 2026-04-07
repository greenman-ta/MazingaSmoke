import torch
import torch.nn as nn

class Selecta(nn.Module):

    def __init__(self, k_max: int = 4, radius: int = 2):
        super().__init__()
        self.k_max = k_max
        self.radius = radius

    def get_topk_indices(self, w):
        """
        Calcola gli indici dei K frame migliori basandosi sui pesi temporali w
        usando una logica greedy
        """
        B, T = w.shape
        K = min(self.k_max, T)
        device = w.device
        
        t = torch.arange(T, device=device)                 
        w_work = w.clone()

        idx_list = []
        for _ in range(K):
            # 1. trova il max attuale
            idx = w_work.argmax(dim=1)                       # [B]
            idx_list.append(idx)
            
            # 2. maschera centro e vicini entro il raggio per evitare frame adiacenti
            gap = self.radius
            mask = (t.unsqueeze(0) - idx.unsqueeze(1)).abs() <= gap
            w_work = w_work.masked_fill(mask, float('-inf'))
            
        idx = torch.stack(idx_list, dim=1)                   # [B,K]
        
        # ordina gli indici temporalmente 
        idx, _ = torch.sort(idx, dim=1)
        return idx

    def forward(self, s, w):
        
        idx = self.get_topk_indices(w) # [B,T] -> [B, K]
        
        # estrai i pesi corrispondenti ai frame scelti
        w_topk = torch.gather(w, 1, idx) # [B, K]

        return idx, w_topk
