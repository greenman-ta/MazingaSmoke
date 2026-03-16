from __future__ import annotations

import os
import random

import numpy as np
import torch
import math



def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch (CPU+CUDA)"""
    seed = int(seed)

    # Python
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

        # CUDA  determinism
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g

def linear_ramp(ep: int, start: int, end: int, v0: float, v1: float) -> float:
    if ep <= start:
        return float(v0)
    if ep >= end:
        return float(v1)
    t = (ep - start) / float(end - start)
    return float(v0 + t * (v1 - v0))

def focal_gamma_schedule(
    ep: int,
    warm_end: int = 4,        # epoche "quasi BCE"
    ramp_end: int = 12,       # fine ramp verso gamma finale
    g0: float = 0.0,          # gamma iniziale (BCE ~ 0)
    g_warm: float = 0.5,      # gamma a fine warmup
    g_final: float = 2.0      # gamma finale
) -> float:

    if ep is None:
        return float(g_final)

    if ep <= warm_end:
        return linear_ramp(ep, 1, warm_end, g0, g_warm)
    else:
        return linear_ramp(ep, warm_end + 1, ramp_end, g_warm, g_final)

def set_loss_gamma(loss_obj, gamma: float) -> bool:
    
    #Setta loss_obj.gamma se esiste. Ritorna True se applicato.
    
    if hasattr(loss_obj, "gamma"):
        loss_obj.gamma = float(gamma)
        return True
    return False


# =========================
# BOOTSTRAP CONFIG
# =========================
BOOT_CFG = {
    "enabled": True,
    "beta": 0.10,         # quanto "ammorbidire" verso p
    "neg_thr": 0.85,      # se y=0 e p>=neg_thr -> sospetto
    "pos_thr": 0.15,      # se y=1 e p<=pos_thr -> sospetto
    "start_epoch": 14,     # da quando parte (epoca 1-based)
    "disable_on_mixup": True,  # non applicare su mixup
}

@torch.no_grad()
def bootstrap_binary_targets(y: torch.Tensor, logits: torch.Tensor, cfg=BOOT_CFG):
    """
    y:      [B] float (0/1)
    logits: [B] float
    Ritorna y_tilde: [B] float in [0,1]
    """
    if (not cfg.get("enabled", True)) or cfg.get("beta", 0.0) <= 0:
        return y.float()

    y = y.float()
    p = torch.sigmoid(logits).detach()

    neg_thr = float(cfg.get("neg_thr", 0.85))
    pos_thr = float(cfg.get("pos_thr", 0.15))
    beta    = float(cfg.get("beta", 0.10))

    m_neg = (y <= 0.0) & (p >= neg_thr)
    m_pos = (y >= 1.0) & (p <= pos_thr)
    m = m_neg | m_pos

    if not torch.any(m):
        return y

    y_tilde = y.clone()
    y_tilde[m] = (1.0 - beta) * y[m] + beta * p[m]
    return y_tilde


def bootstrap_targets(y: torch.Tensor, logits: torch.Tensor, *, ep: int, use_mixup: bool, cfg=BOOT_CFG):
    """
    Wrapper che decide se applicare bootstrap.
    ep: epoca 1-based
    """
    if not cfg.get("enabled", True):
        return y.float()

    if cfg.get("disable_on_mixup", True) and use_mixup:
        return y.float()

    start_ep = int(cfg.get("start_epoch", 2))
    if ep is None or ep < start_ep:
        return y.float()

    return bootstrap_binary_targets(y, logits, cfg=cfg)