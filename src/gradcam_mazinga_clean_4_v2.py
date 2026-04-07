import torch
import torch.nn.functional as Fnn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Union
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

TARGET_LAYER = "backbone.proj_deep"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _resolve_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    cur = model
    for part in name.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def _denorm_img(t: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    m = torch.tensor(mean, dtype=t.dtype, device=t.device)[:, None, None]
    s = torch.tensor(std, dtype=t.dtype, device=t.device)[:, None, None]
    x = (t * s + m).clamp(0, 1)
    return x.permute(1, 2, 0).detach().cpu().numpy()


def _resize_cam(cam: np.ndarray, target_hw) -> np.ndarray:
    H, W = target_hw
    cam_t = torch.from_numpy(cam)[None, None, ...].float()
    cam_r = Fnn.interpolate(cam_t, size=(H, W), mode="bilinear", align_corners=False)
    return cam_r[0, 0].cpu().numpy()


def _overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cm.jet(cam)[..., :3]
    out = (1 - alpha) * rgb + alpha * heat
    return np.clip(out, 0, 1)


def _normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m, M = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(m) or not np.isfinite(M) or M <= m:
        return np.zeros_like(x, dtype=np.float32)
    return (x - m) / (M - m + 1e-6)


def vis_saliency_5rows_singleclip(
    model: torch.nn.Module,
    rgb_clip: torch.Tensor,                    # [T,3,H,W]
    weights_1d: Union[torch.Tensor, np.ndarray],
    save_path: Union[str, Path],
    topk_idx: List[int],
    post_attn_maps_k: Optional[Union[np.ndarray, torch.Tensor]] = None,
    topk_order: Optional[Union[np.ndarray, torch.Tensor, List[int]]] = None,
    w_eff: Optional[Union[np.ndarray, torch.Tensor]] = None,
    target_layer_backbone: str = TARGET_LAYER,
    max_frames: int = 6,
    device=None,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    idx_np = topk_idx.detach().cpu().numpy() if isinstance(topk_idx, torch.Tensor) else np.asarray(topk_idx)
    idx_np = np.asarray(idx_np, dtype=int).reshape(-1)

    if len(idx_np) == 0:
        return

    frame_to_kpos = {int(f): i for i, f in enumerate(idx_np.tolist())}

    device = next(model.parameters()).device if device is None else device

    model.eval()
    x = rgb_clip.unsqueeze(0).to(device)  # [1,T,3,H,W]
    x.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = model(x)
    score = logits[0] if logits.dim() == 1 else logits[0, 0]
    score.backward()

    # FUSION BACKWARD CAM
    fm = model.fusion_module.last_feat          # [B,K,C,Hf,Wf]
    gm = fm.grad
    F_fus = fm[0]
    G_fus = gm[0]

    alpha = G_fus.mean(dim=(2, 3))                                   # [K,C]
    cams_k = torch.relu((F_fus * alpha[..., None, None]).sum(1))     # [K,Hf,Wf]
    cams_k = cams_k - cams_k.amin(dim=(1, 2), keepdim=True)
    cams_k = cams_k / (cams_k.amax(dim=(1, 2), keepdim=True) + 1e-8)
    cams_k_np = cams_k.detach().cpu().numpy()

    if topk_order is not None:
        if isinstance(topk_order, torch.Tensor):
            order = topk_order.detach().cpu().long().tolist()
        else:
            order = list(np.asarray(topk_order).astype(int).tolist())
        if len(order) > 0:
            order = [o for o in order if 0 <= o < cams_k_np.shape[0]]
            if len(order) > 0:
                cams_k_np = cams_k_np[order]

    # SCORE COMBINATO per scegliere i frame
    # temporal + fusion + cam
    w_np = weights_1d.detach().cpu().numpy() if isinstance(weights_1d, torch.Tensor) else np.asarray(weights_1d)
    w_np = np.asarray(w_np, dtype=np.float32).reshape(-1)

    cam_strength = cams_k.mean(dim=(1, 2)).detach().cpu().numpy()   
    cam_strength = np.asarray(cam_strength, dtype=np.float32).reshape(-1)

    n = min(len(idx_np), len(cam_strength))
    idx_np = idx_np[:n]
    cam_strength = cam_strength[:n]
    cams_k_np = cams_k_np[:n]

    temporal_sel = np.zeros(n, dtype=np.float32)
    for i, t in enumerate(idx_np):
        if 0 <= t < len(w_np):
            temporal_sel[i] = w_np[t]

    temporal_norm = _normalize_01(temporal_sel)
    cam_norm = _normalize_01(cam_strength)

    if w_eff is not None:
        eff_np = w_eff.detach().cpu().numpy() if isinstance(w_eff, torch.Tensor) else np.asarray(w_eff)
        eff_np = np.asarray(eff_np, dtype=np.float32).reshape(-1)[:n]
        fusion_norm = _normalize_01(eff_np)

        vis_score = 0.50 * temporal_norm + 0.35 * fusion_norm + 0.15 * cam_norm
    else:
        vis_score = 0.65 * temporal_norm + 0.35 * cam_norm

    sel = np.argsort(-vis_score)[:max_frames]
    frame_idx = sorted(idx_np[sel].tolist())

    K_show = len(frame_idx)
    if K_show == 0:
        return

    # PLOT
    fig = plt.figure(figsize=(2.55 * K_show, 7.9))

    outer = gridspec.GridSpec(
        3, 1,
        height_ratios=[3.2, 1.9, 3.2],
        hspace=0.28
    )

    gs_top = outer[0].subgridspec(1, K_show, wspace=0.0, hspace=0.0)
    gs_bot = outer[2].subgridspec(1, K_show, wspace=0.0, hspace=0.0)

    imgs_cache = {}

    # Riga 1: originali
    for ci, t in enumerate(frame_idx):
        ax = fig.add_subplot(gs_top[0, ci])
        img = _denorm_img(rgb_clip[t])
        imgs_cache[t] = img

        k_idx = frame_to_kpos.get(int(t), ci)
        temp_v = temporal_sel[k_idx] if 0 <= k_idx < len(temporal_sel) else 0.0
        cam_v  = cam_strength[k_idx] if 0 <= k_idx < len(cam_strength) else 0.0
        vis_v  = vis_score[k_idx] if 0 <= k_idx < len(vis_score) else 0.0

        ax.imshow(img)
        ax.axis("off")
        ax.set_aspect("auto")
        ax.set_title(f"{t}\nS={vis_v:.3f} T={temp_v:.3f} C={cam_v:.3f}", fontsize=9, pad=6)

    # Riga 2: grafico pesi
    axw = fig.add_subplot(outer[1])

    x_all = np.arange(len(w_np))
    axw.plot(x_all, w_np, marker="o", linewidth=1.5, label="temporal")

    if w_eff is not None:
        eff = w_eff.detach().cpu().numpy() if isinstance(w_eff, torch.Tensor) else np.asarray(w_eff)
        eff = np.asarray(eff, dtype=np.float32)
        idx = np.asarray(topk_idx, dtype=int)
        w_eff_T = np.zeros_like(w_np, dtype=np.float32)
        for k, t in enumerate(idx):
            if k < len(eff) and 0 <= t < len(w_eff_T):
                w_eff_T[t] = eff[k]
        axw.plot(np.arange(len(w_eff_T)), w_eff_T, marker="x", linestyle="--", linewidth=1.2, label="w_eff")
        axw.legend(loc="upper right", fontsize=9)

    ymin, ymax = axw.get_ylim()
    for t in frame_idx:
        if 0 <= t < len(w_np):
            axw.vlines(t, ymin, ymax, linestyles="dotted", linewidth=0.8, alpha=0.55)

    axw.set_xlim(-0.5, max(len(w_np) - 0.5, 0.5))
    axw.set_xlabel("frame", labelpad=8)
    axw.set_ylabel("peso", labelpad=10)
    axw.grid(True, alpha=0.35)
    axw.margins(x=0.01, y=0.08)
    axw.set_xticks(np.arange(len(w_np)))
    axw.set_xticklabels([str(i) for i in range(len(w_np))], fontsize=9)
    axw.tick_params(axis="x", pad=6)
    axw.tick_params(axis="y", pad=4)

    # Riga 3: fusion backward
    for ci, t in enumerate(frame_idx):
        ax = fig.add_subplot(gs_bot[0, ci])
        img = imgs_cache[t]

        k_idx = frame_to_kpos.get(int(t), ci)
        if k_idx >= cams_k_np.shape[0]:
            k_idx = cams_k_np.shape[0] - 1

        cam_resized = _resize_cam(cams_k_np[k_idx], img.shape[:2])
        ax.imshow(_overlay_cam(img, cam_resized, alpha=0.45))
        ax.axis("off")
        ax.set_aspect("auto")

    fig.subplots_adjust(
        left=0.0,
        right=1.0,
        top=0.985,
        bottom=0.06
    )

    fig.savefig(str(save_path), dpi=140, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)