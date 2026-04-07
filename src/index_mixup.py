import torch
import torch.nn.functional as Fnn

def build_mixup_index(labels01: torch.Tensor, mode: str):
    """
    Costruisce un indice di mixup per batch bilanciato.
    mode: "hetero" oppure "homo"
    """
    B = labels01.numel()
    device = labels01.device
    lab = (labels01 >= 0.5).long()

    idx_pos = torch.where(lab == 1)[0]
    idx_neg = torch.where(lab == 0)[0]

    # batch monoclasse -> permutazione casuale
    if idx_pos.numel() == 0 or idx_neg.numel() == 0:
        return torch.randperm(B, device=device)

    index = torch.empty(B, dtype=torch.long, device=device)

    if mode == "hetero":
        pick_neg = idx_neg[torch.randint(0, idx_neg.numel(), (idx_pos.numel(),), device=device)]
        pick_pos = idx_pos[torch.randint(0, idx_pos.numel(), (idx_neg.numel(),), device=device)]
        index[idx_pos] = pick_neg
        index[idx_neg] = pick_pos
    else:
        pick_pos = idx_pos[torch.randint(0, idx_pos.numel(), (idx_pos.numel(),), device=device)]
        pick_neg = idx_neg[torch.randint(0, idx_neg.numel(), (idx_neg.numel(),), device=device)]
        index[idx_pos] = pick_pos
        index[idx_neg] = pick_neg

    # evita self-pairing
    same = (index == torch.arange(B, device=device))
    if same.any():
        rp = torch.randperm(B, device=device)
        index[same] = rp[same]

    return index