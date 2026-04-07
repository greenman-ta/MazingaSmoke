import torch
import torch.nn.functional as Fnn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Dict, Union
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

    # sceglie i frame da mostrare 
    frame_idx = list(topk_idx)[:max_frames]
    K_show = len(frame_idx)
    if K_show == 0:
        return

    if isinstance(post_attn_maps_k, torch.Tensor):
        post_attn_maps_k = post_attn_maps_k.detach().cpu().numpy()
    post_attn_maps_k = np.asarray(post_attn_maps_k)

    # device 
    device = next(model.parameters()).device if device is None else device


    # 1) BACKWARD UNICO + HOOK BACKBONE 
    target_module = _resolve_module_by_name(model, target_layer_backbone)
    F_hook, G_hook = None, None #f -> feature per layer, g -> grad per layer

    def fwd_hook(m, inp, out):
        nonlocal F_hook
        F_hook = out

    def bwd_hook(m, grad_input, grad_output):
        nonlocal G_hook
        G_hook = grad_output[0]

    #salvo gradienti e feature
    h1 = target_module.register_forward_hook(fwd_hook)
    h2 = target_module.register_full_backward_hook(bwd_hook)

    model.eval()
    x = rgb_clip.unsqueeze(0).to(device)  # [1,T,3,H,W]
    x.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = model(x)
    score = logits[0] if logits.dim() == 1 else logits[0, 0]
    score.backward()

    h1.remove()
    h2.remove()

    def cam_single(feat_chtw: torch.Tensor, grad_chtw: torch.Tensor) -> np.ndarray:
        w = grad_chtw.view(grad_chtw.shape[0], -1).mean(dim=1)     # [C]
        cam = torch.einsum("chw,c->hw", feat_chtw.float(), w.float())
        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()
        m, M = cam.min(), cam.max()
        return (cam - m) / (M - m + 1e-6) if M > m else np.zeros_like(cam, np.float32)

    # backbone cam
    cams_back = {}
    T = rgb_clip.shape[0]
    b_index = 0
    for t in frame_idx:
        bt = b_index * T + t #trovo indice corretto nel batch
        cams_back[t] = cam_single(F_hook[bt], G_hook[bt])   

    # POST-FUSION BACKWARD CAM 
    fm = model.fusion_module.last_feat          # [B,K,C,Hf,Wf]
    gm = fm.grad                               
    F_fus = fm[0]
    G_fus = gm[0]

    # gradcam per K
    alpha = G_fus.mean(dim=(2, 3))                                   # [K,C]
    cams_k = torch.relu((F_fus * alpha[..., None, None]).sum(1))     # [K,Hf,Wf]
    #normalizza
    cams_k = cams_k - cams_k.amin(dim=(1, 2), keepdim=True)
    cams_k = cams_k / (cams_k.amax(dim=(1, 2), keepdim=True) + 1e-8)
    cams_k_np = cams_k.detach().cpu().numpy()
    #ordina
    order = topk_order.detach().cpu().long().tolist()
    cams_k_np = cams_k_np[order]

    # STATIC BRANCH CAM 
    sb_fm = model.static_branch.last_feat
    sb_gm = sb_fm.grad
    if sb_fm.dim() == 5:
        sb_fm = sb_fm[0]
        sb_gm = sb_gm[0]
    alpha_s = sb_gm.mean(dim=(2, 3))
    cams_s = torch.relu((sb_fm * alpha_s[..., None, None]).sum(1))
    # normalizza
    cams_s = cams_s - cams_s.amin(dim=(1, 2), keepdim=True)
    cams_s = cams_s / (cams_s.amax(dim=(1, 2), keepdim=True) + 1e-8)
    cams_s_np = cams_s.detach().cpu().numpy()

    # PLOT: 6 righe
    fig = plt.figure(figsize=(3 * K_show, 12.5))
    gs = gridspec.GridSpec(6, K_show, height_ratios=[3, 1.2, 3, 3, 3, 3])

    # Riga 1: originali
    imgs_cache = {}
    for ci, t in enumerate(frame_idx):
        ax = fig.add_subplot(gs[0, ci])
        img = _denorm_img(rgb_clip[t])
        imgs_cache[t] = img
        ax.imshow(img)
        ax.set_title(f"t={t}")
        ax.axis("off")

    # Riga 2: grafico pesi attenzioni
    axw = fig.add_subplot(gs[1, :])
    w_np = weights_1d.detach().cpu().numpy() if isinstance(weights_1d, torch.Tensor) else np.asarray(weights_1d)
    axw.plot(np.arange(len(w_np)), w_np, marker="o", linewidth=1.5)
    if w_eff is not None:
        eff = w_eff.detach().cpu().numpy() if isinstance(w_eff, torch.Tensor) else np.asarray(w_eff)
        idx = np.asarray(topk_idx, dtype=int)
        w_eff_T = np.zeros_like(w_np, dtype=np.float32)
        for k, t in enumerate(idx):
            if 0 <= t < len(w_eff_T):
                w_eff_T[t] = eff[k]
        axw.plot(np.arange(len(w_eff_T)), w_eff_T, marker="x", linestyle="--", linewidth=1.2)

    ymin, ymax = axw.get_ylim()
    for t in frame_idx:
        if 0 <= t < len(w_np):
            axw.vlines(t, ymin, ymax, linestyles="dotted", linewidth=0.8, alpha=0.5)

    axw.set_xlim(-0.5, max(len(w_np) - 0.5, 0.5))
    axw.set_xlabel("frame")
    axw.set_ylabel("peso")
    axw.grid(True)
    axw.set_title(f"Dist attenzioni={len(w_np)}")

    # Row 3: backbone 
    cams_resized_cache = {}
    for ci, t in enumerate(frame_idx):
        ax = fig.add_subplot(gs[2, ci])
        img = imgs_cache[t]
        cam_resized = _resize_cam(cams_back[t], img.shape[:2])
        cams_resized_cache[t] = cam_resized
        ax.imshow(_overlay_cam(img, cam_resized, alpha=0.45))
        ax.set_title("Backbone")
        ax.axis("off")

    # Row 4: static branch
    for ci, t in enumerate(frame_idx):
        ax = fig.add_subplot(gs[3, ci])
        img = imgs_cache[t]
        k_idx = ci if ci < cams_s_np.shape[0] else (cams_s_np.shape[0] - 1)
        cam_resized = _resize_cam(cams_s_np[k_idx], img.shape[:2])
        ax.imshow(_overlay_cam(img, cam_resized, alpha=0.45))
        ax.set_title("Static branch")
        ax.axis("off")

    # Row 5: fusion-forward
    for ci, t in enumerate(frame_idx):
        ax = fig.add_subplot(gs[4, ci])
        img = imgs_cache[t]
        cam_back = cams_resized_cache[t]
        att_small = post_attn_maps_k[ci] if ci < len(post_attn_maps_k) else np.zeros((7, 7), np.float32)
        att_resized = _resize_cam(att_small, img.shape[:2])
        ax.imshow(_overlay_cam(img, _normalize_01(att_resized), alpha=0.45))
        ax.set_title("Fusion forward")
        ax.axis("off")

    # Row 6: fusion-backward
    for ci, t in enumerate(frame_idx):
        ax = fig.add_subplot(gs[5, ci])
        img = imgs_cache[t]
        k_idx = ci if ci < cams_k_np.shape[0] else (cams_k_np.shape[0] - 1)
        cam_resized = _resize_cam(cams_k_np[k_idx], img.shape[:2])
        ax.imshow(_overlay_cam(img, cam_resized, alpha=0.45))
        ax.set_title("Fusion backward")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)