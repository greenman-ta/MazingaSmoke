import torch
import torch.nn.functional as Fnn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Dict, Union

# --- Hook layers ---
TARGET_LAYER = "backbone.proj_deep"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _resolve_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    cur = model
    for part in name.split("."):
        if part.isdigit(): cur = cur[int(part)]
        else: cur = getattr(cur, part)
    return cur

class GradCAMLite:
    def __init__(self, model: torch.nn.Module, target_layer_name: str, device=None):
        self.model = model
        self.device = device
        self.target_layer_name = target_layer_name
        self.target_module = _resolve_module_by_name(model, target_layer_name)

        self._F = None
        self._dF = None
        self._h_fwd = None
        self._h_bwd = None

    def __enter__(self):
        def fwd_hook(module, inp, out):
            self._F = out
        def bwd_hook(module, grad_input, grad_output):
            self._dF = grad_output[0]

        self._h_fwd = self.target_module.register_forward_hook(fwd_hook)
        self._h_bwd = self.target_module.register_full_backward_hook(bwd_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._h_fwd is not None:
            self._h_fwd.remove(); self._h_fwd = None
        if self._h_bwd is not None:
            self._h_bwd.remove(); self._h_bwd = None
        # PULIZIA ESPLICITA DELLA MEMORIA
        self._F = None
        self._dF = None

    @staticmethod
    def _compute_cam_single(feat: torch.Tensor, grad: torch.Tensor) -> np.ndarray:
        feat = feat.float().contiguous()
        grad = grad.float().contiguous()
        w = grad.view(grad.shape[0], -1).mean(dim=1)         # [C]
        cam = torch.einsum('chw,c->hw', feat, w)             # [H,W]
        cam = torch.relu(cam)
        cam_np = cam.detach().cpu().numpy()
        m, M = cam_np.min(), cam_np.max()
        if np.isfinite(m) and np.isfinite(M) and M > m:
            cam_np = (cam_np - m) / (M - m + 1e-6)
        else:
            cam_np = np.zeros_like(cam_np, dtype=np.float32)
        return cam_np

    def generate(self, rgb_5d: torch.Tensor, frames_to_viz: Optional[List[int]] = None, b_index: int = 0) -> Dict[int, np.ndarray]:
        self.model.eval()
        device = next(self.model.parameters()).device if self.device is None else self.device
        x = rgb_5d.unsqueeze(0).to(device) if rgb_5d.dim() == 4 else rgb_5d.to(device)
        
        # CRUCIALE per backprop fino al backbone
        x.requires_grad_(True)

        B_in, T_in = x.shape[0], x.shape[1]
        self.model.zero_grad(set_to_none=True)
        
        with torch.enable_grad():
            logits = self.model(x)
            score = logits[b_index] if logits.dim() == 1 else logits[b_index, 0]
            if not score.requires_grad:
                raise RuntimeError("Logit has no grad.")
            score.backward()

        if frames_to_viz is None:
            frames_to_viz = list(range(T_in))
        cams: Dict[int, np.ndarray] = {}

        # Safety check
        if self._F is None or self._dF is None:
             # Se mancano hook, ritorna vuoto o solleva errore gestibile
             print("[VIS][WARN] Hooks did not capture features/grads.")
             return {t: np.zeros((7,7), dtype=np.float32) for t in frames_to_viz}

        if self._F.dim() == 5:
            for t in frames_to_viz:
                F_bt = self._F[b_index, t]
                G_bt = self._dF[b_index, t]
                cams[t] = self._compute_cam_single(F_bt, G_bt)
        elif self._F.dim() == 4:
            N = self._F.shape[0]
            if B_in == 1 and N == T_in:
                for t in frames_to_viz:
                    cams[t] = self._compute_cam_single(self._F[t], self._dF[t])
            else:
                F_b, G_b = self._F[b_index], self._dF[b_index]
                cam_one = self._compute_cam_single(F_b, G_b)
                for t in frames_to_viz:
                    cams[t] = cam_one
        else:
            raise ValueError(f"Unsupported feature shape for GradCAMLite: {self._F.shape}")
        return cams

# --- helpers ---
def _denorm_img(t: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    m = torch.tensor(mean, dtype=t.dtype, device=t.device)[:, None, None]
    s = torch.tensor(std, dtype=t.dtype, device=t.device)[:, None, None]
    x = (t * s + m).clamp(0, 1)
    return x.permute(1, 2, 0).detach().cpu().numpy()

def _resize_cam(cam: np.ndarray, target_hw) -> np.ndarray:
    H, W = target_hw
    cam_t = torch.from_numpy(cam)[None, None, ...].float()
    cam_r = Fnn.interpolate(cam_t, size=(H, W), mode='bilinear', align_corners=False)
    return cam_r[0, 0].cpu().numpy()

def _overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib.cm as cm
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
    rgb_clip: torch.Tensor,                   
    weights_1d: torch.Tensor,                 
    save_path: Path,
    target_layer_backbone: str = TARGET_LAYER,
    max_frames: int = 6,
    device=None,
    topk_idx: Optional[List[int]] = None,                 
    post_attn_maps_k: Optional[Union[np.ndarray, torch.Tensor]] = None,   
    topk_order: Optional[Union[np.ndarray, torch.Tensor, List[int]]] = None,  
    display_mode_row5: str = "overlay",       
    w_eff: Optional[Union[np.ndarray, torch.Tensor]] = None
):
    save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ---- 1) Forward+backward con Context Manager (PROTEZIONE CRASH) ----
    cams_back = {}
    try:
        # Il blocco 'with' garantisce che cam_engine.close() venga chiamato 
        # anche se generate() lancia un errore o un OOM.
        with GradCAMLite(model, target_layer_backbone, device=device) as cam_engine:
            cams_back = cam_engine.generate(rgb_clip.unsqueeze(0), frames_to_viz=topk_idx, b_index=0)
    except Exception as e:
        print(f"[VIS][ERROR] GradCAMLite backbone failed: {e}")
        # Fallback per non rompere la visualizzazione completa
        cams_back = {t: np.zeros((7,7), dtype=np.float32) for t in (topk_idx if topk_idx else [])}

    # ---- 3) Prepare images ----
    if topk_idx is None: topk_idx = []
    frame_idx = list(topk_idx)[:max_frames]
    K_show = len(frame_idx)
    
    # Gestione fallback se K_show è 0
    if K_show == 0: 
        return

    if post_attn_maps_k is not None:
        if isinstance(post_attn_maps_k, torch.Tensor):
            post_attn_maps_k = post_attn_maps_k.detach().cpu().numpy()
        post_attn_maps_k = np.asarray(post_attn_maps_k) 
        if post_attn_maps_k.ndim == 4 and post_attn_maps_k.shape[1] == 1:
            post_attn_maps_k = post_attn_maps_k[:,0]
    else:
        post_attn_maps_k = np.zeros((K_show, 7, 7), dtype=np.float32)

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(3*K_show, 12.5))
    gs = gridspec.GridSpec(6, K_show, height_ratios=[3, 1.2, 3, 3, 3, 3])

    # Row 1
    imgs_cache = {}
    for ci, t in enumerate(frame_idx):
        ax = fig.add_subplot(gs[0, ci])
        img = _denorm_img(rgb_clip[t])
        imgs_cache[t] = img
        ax.imshow(img); ax.set_title(f"t={t}"); ax.axis("off")

    # Row 2
    axw = fig.add_subplot(gs[1, :])
    w_np = weights_1d.detach().cpu().numpy() if isinstance(weights_1d, torch.Tensor) else np.asarray(weights_1d)
    axw.plot(np.arange(len(w_np)), w_np, marker='o', linewidth=1.5)
    if (w_eff is not None) and (topk_idx is not None):
        eff = w_eff.detach().cpu().numpy() if isinstance(w_eff, torch.Tensor) else np.asarray(w_eff)
        idx = np.asarray(topk_idx, dtype=int)
        w_eff_T = np.zeros_like(w_np, dtype=np.float32)
        for k, t in enumerate(idx):
            if 0 <= t < len(w_eff_T): w_eff_T[t] = eff[k]
        axw.plot(np.arange(len(w_eff_T)), w_eff_T, marker='x', linestyle='--', linewidth=1.2,)

    ymin, ymax = axw.get_ylim()
    for t in frame_idx:
        if 0 <= t < len(w_np): axw.vlines(t, ymin, ymax, linestyles='dotted', linewidth=0.8, alpha=0.5)
    axw.set_xlim(-0.5, max(len(w_np)-0.5, 0.5))
    axw.set_xlabel("frame"); axw.set_ylabel("peso"); axw.grid(True); 
    axw.set_title(f"Dist attenzioni={len(w_np)}")

    # Row 3
    cams_resized_cache = {}
    for ci, t in enumerate(frame_idx):
        ax = fig.add_subplot(gs[2, ci])
        img = imgs_cache[t]
        cam_small = cams_back.get(t, np.zeros((7,7))) 
        cam_resized = _resize_cam(cam_small, img.shape[:2])
        cams_resized_cache[t] = cam_resized
        ax.imshow(_overlay_cam(img, cam_resized, alpha=0.45))
        ax.set_title("Grad-CAM (backbone)"); ax.axis("off")

    # Row 4
    _row_static_branch_cam(fig, gs, 3, imgs_cache, frame_idx, model)

    # Row 5
    for ci, t in enumerate(frame_idx):
        ax = fig.add_subplot(gs[4, ci])
        img = imgs_cache[t]
        k_idx = ci
        cam_back = cams_resized_cache[t]
        if k_idx < len(post_attn_maps_k):
            att_small = post_attn_maps_k[k_idx]
            att_resized = _resize_cam(att_small, img.shape[:2])
        else:
            fused = np.zeros_like(cam_back)
        
        ax.imshow(_overlay_cam(img, _normalize_01(att_resized), alpha=0.45))
        ax.set_title("Teacher attn_mapK (overlay)"); ax.axis("off")

    # Row 6
    _row5_true_postfusion(fig, gs, 5, imgs_cache, frame_idx, model, display_mode_row5, topk_order)

    save_path = Path(save_path)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)


def _row5_true_postfusion(fig, gs, row_idx, imgs_cache, frame_idx, model, display_mode_row5="overlay", topk_order=None):
    fm = getattr(model.fusion_module, "last_feat", None)
    if (fm is None) or (fm.grad is None):
        for ci, t in enumerate(frame_idx):
            ax = fig.add_subplot(gs[row_idx, ci])
            ax.imshow(imgs_cache[t])
            ax.text(0.5, 0.5, "no grad", ha="center", va="center", transform=ax.transAxes, fontsize=8, bbox=dict(fc="white", alpha=0.7, ec="none"))
            ax.set_title("True Post-fusion Grad-CAM"); ax.axis("off")
        return

    F = fm[0]; G = fm.grad[0]
    alpha = G.mean(dim=(2, 3))
    cams_k = torch.relu((F * alpha[..., None, None]).sum(1))
    cams_k = cams_k.detach()
    cams_k = cams_k - cams_k.amin(dim=(1, 2), keepdim=True)
    cams_k = cams_k / (cams_k.amax(dim=(1, 2), keepdim=True) + 1e-8)

    if topk_order is not None:
        if isinstance(topk_order, torch.Tensor): order = topk_order.detach().cpu().long().tolist()
        else: order = list(topk_order)
        if len(order) == cams_k.shape[0]: cams_k = cams_k[order]

    cams_k_np = cams_k.detach().cpu().numpy()

    for ci, t in enumerate(frame_idx):
        ax = fig.add_subplot(gs[row_idx, ci])
        img = imgs_cache[t]
        k_idx = ci if ci < cams_k_np.shape[0] else (cams_k_np.shape[0] - 1)
        cam_true = cams_k_np[k_idx]
        cam_resized = _resize_cam(cam_true, img.shape[:2])

        if display_mode_row5 == "heatmap": ax.imshow(cam_resized, vmin=0, vmax=1, cmap='jet')
        else: ax.imshow(_overlay_cam(img, cam_resized, alpha=0.45))
        
        ax.set_title("True Post-fusion Grad-CAM"); ax.axis("off")

def _row_static_branch_cam(fig, gs, row_idx, imgs_cache, frame_idx, model):
    sb = getattr(model, "static_branch", None)
    if sb is None or not hasattr(sb, "last_feat") or sb.last_feat is None:
        for ci, t in enumerate(frame_idx):
            ax = fig.add_subplot(gs[row_idx, ci])
            ax.imshow(imgs_cache[t])
            ax.text(0.5, 0.5, "no static_feat", ha="center", va="center", transform=ax.transAxes, fontsize=8, bbox=dict(fc="white", alpha=0.7, ec="none"))
            ax.set_title("Static branch CAM"); ax.axis("off")
        return

    fm = sb.last_feat
    grad = fm.grad if hasattr(fm, "grad") else None
    if grad is None:
        for ci, t in enumerate(frame_idx):
            ax = fig.add_subplot(gs[row_idx, ci])
            ax.imshow(imgs_cache[t])
            ax.text(0.5, 0.5, "no grad", ha="center", va="center", transform=ax.transAxes, fontsize=8, bbox=dict(fc="white", alpha=0.7, ec="none"))
            ax.set_title("Static branch CAM"); ax.axis("off")
        return

    if fm.dim() == 5: fm = fm[0]; grad = grad[0]
    elif fm.dim() != 4: return

    alpha = grad.mean(dim=(2, 3))
    cams_k = torch.relu((fm * alpha[..., None, None]).sum(1))
    cams_k = cams_k.detach()
    cams_k = cams_k - cams_k.amin(dim=(1, 2), keepdim=True)
    cams_k = cams_k / (cams_k.amax(dim=(1, 2), keepdim=True) + 1e-8)
    cams_k_np = cams_k.cpu().numpy()

    for ci, t in enumerate(frame_idx):
        ax = fig.add_subplot(gs[row_idx, ci])
        img = imgs_cache[t]
        k_idx = ci if ci < cams_k_np.shape[0] else (cams_k_np.shape[0] - 1)
        cam_s = cams_k_np[k_idx]
        cam_resized = _resize_cam(cam_s, img.shape[:2])
        ax.imshow(_overlay_cam(img, cam_resized, alpha=0.45))
        ax.set_title("Static branch Grad-CAM"); ax.axis("off")