
import os, argparse
import torch
from torch.optim import AdamW
from sys import path
from torch.utils.data import DataLoader, Dataset
from torch import nn
from datetime import datetime
from tqdm import tqdm
import numpy as np
from timm.optim import Lookahead
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts,LinearLR, ReduceLROnPlateau
from src.util_2 import focal_gamma_schedule, set_loss_gamma, bootstrap_targets


# MODEL 
from mazinga_smoke.mazinga_smoke import MazingaSmokeClassifier

from src.metrics_utils import compute_and_log_metrics
from src.focal_loss import BinaryFocalLossBalanced
from src.load_dataset import loadDataset
from src.index_mixup import build_mixup_index
from src.manage_checkpoint import (load_model_weights_only,load_full_checkpoint, save_full_checkpoint, default_meta)
from src.run_policy import (resolve_resume_policy,apply_finetune_lr,apply_finetune_freeze)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F
from src.base_learner_3_f import BaseLearner

# CONFIG
DEFAULT_EPOCHS = 50
DEFAULT_LR = 5e-4
DEFAULT_BATCH = 18
DEFAULT_WORKERS = 4
DEFAULT_NUM_SEGMENTS = 36
DEFAULT_RGB_DIR = "/rise/deep-smoke-machine/back-end/data/rgb224"
THR = 0.5

N_VIS = 40
MAX_PER_CLASS = 10

FINETUNE_INIT_WEIGHTS = ""

run_dir_global = "."

#funzione per calcolare alpha_pos e alpha_neg per focal loss bilanciata
def compute_focal_alphas(train_ds):

    n_pos, n_neg = train_ds.get_label_counts()
    total = n_pos + n_neg
    eps = 1e-6

    alpha_pos = total / (2.0 * max(n_pos, 1))
    alpha_neg = total / (2.0 * max(n_neg, 1))

    print(f"[FOCAL] Class balance train (dopo patch): pos={n_pos} neg={n_neg} tot={total}")
    print(f"[FOCAL] alpha_pos={alpha_pos:.3f}  alpha_neg={alpha_neg:.3f}")

    return float(alpha_pos), float(alpha_neg)
    
class TransformHelper(BaseLearner):
    def fit(self): pass
    def test(self): pass

helper = TransformHelper(use_cuda=False)


# TRAIN

def train_loop(model, loader, optimizer, device, ce_loss_fn, ce_loss_mix, DO_MIXUP, log_interval=20, ep=None):
    global run_dir_global
    
    model.train()
    total_loss = 0.0
    n_samples = 0
    iterator = tqdm(enumerate(loader), total=len(loader), desc="train", leave=True, dynamic_ncols=True)
    y_true, y_pred, y_score = [], [], []

    # --- CONFIGURAZIONE MIXUP ---

    # schedule
    MIXUP_ALPHA = 0.4          
    MIXUP_END   = 12           

    if ep <= 4:
        MIXUP_PROB = 0.40      
        P_HETERO   = 0.30      
    elif ep <= MIXUP_END:
        MIXUP_PROB = 0.15      
        P_HETERO   = 0.00
    else:
        MIXUP_PROB = 0.00
        P_HETERO   = 0.00
    
    # ---- FOCAL ----
    g_clean = focal_gamma_schedule(ep, warm_end=4, ramp_end=12, g0=0.0, g_warm=0.5, g_final=2.0)
    set_loss_gamma(ce_loss_fn, g_clean)
    set_loss_gamma(ce_loss_mix, 0.5)

    if (ep == 1):
        tqdm.write(f"[FOCAL] gamma_clean={g_clean:.3f} gamma_mix={getattr(ce_loss_mix,'gamma', None)}")

    for b_idx, batch in iterator:
        rgb = batch["rgb"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).float()
        
        weights = batch.get("weight", torch.ones_like(labels)).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        B = rgb.size(0)

        use_mixup = DO_MIXUP and (B > 1) and (np.random.random() < MIXUP_PROB)

        if use_mixup:
            is_hetero = (np.random.random() < P_HETERO)
            mode = "hetero" if is_hetero else "homo"
            index = build_mixup_index(labels, mode=mode)

            lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)

            if is_hetero:
                lam = 0.75 + 0.25 * lam   

            input_var = lam * rgb + (1 - lam) * rgb[index]

            label_a = labels
            label_b = labels[index]
            labels_mix = lam * label_a + (1 - lam) * label_b

            weight_a = weights
            weight_b = weights[index]
            weight_mixed = lam * weight_a + (1 - lam) * weight_b
        else:
            input_var = rgb
            labels_mix = labels
            weight_mixed = weights


        output = model(input_var)

        # Gestione output

        if isinstance(output, tuple):
            if len(output) == 3:
                main_logits, aux_logits, static_logits = output
            elif len(output) == 2:
                main_logits, aux_logits = output
                static_logits = None
            else:
                main_logits = output[0]
                aux_logits = None
                static_logits = None
        else:
            main_logits = output
            aux_logits = None
            static_logits = None

        # LOSS PRINCIPALE
        loss_fn = ce_loss_mix if use_mixup else ce_loss_fn

        if use_mixup:
            y_main = labels_mix
        else:
            y_main = bootstrap_targets(labels, main_logits, ep=ep, use_mixup=use_mixup)

        loss_raw  = loss_fn(main_logits, y_main)
        loss_main = (loss_raw * weight_mixed).mean()


        # LOSS AUSILIARIA (TEMPORALE)
        loss_aux = 0.0

        if aux_logits is not None:
            if use_mixup:
                y_aux = labels_mix
            else:
                y_aux = bootstrap_targets(labels, aux_logits, ep=ep, use_mixup=use_mixup)

            loss_aux_raw = loss_fn(aux_logits, y_aux)
            loss_aux = (loss_aux_raw * weight_mixed).mean()

        aux_weight = 1.0

        # LOSS STATICA 
        loss_static = 0.0
        lambda_static = 1.0  

        if static_logits is not None:
            if use_mixup:
                y_stat = labels_mix
            else:
                y_stat = bootstrap_targets(labels, static_logits, ep=ep, use_mixup=use_mixup)

            loss_static_raw = loss_fn(static_logits, y_stat)
            loss_static = (loss_static_raw * weight_mixed).mean()

        # LOSS TOTALE
        loss = loss_main + aux_weight * loss_aux + lambda_static * loss_static

        # Debug log
        if ep == 1 and b_idx == 0:
            probs_dbg = torch.sigmoid(main_logits.detach().float())
            tqdm.write(f"\n[DEBUG] ---- EPOCA 1 / BATCH 0 ----") 
            tqdm.write(f"MixUp: {use_mixup}")
            tqdm.write(f"Weights Sample (primi 4): {weight_mixed[:4].cpu().numpy()}")
            tqdm.write(f"Logits: {main_logits[:4].detach().cpu().numpy().round(3)}")
            tqdm.write(f"loss_main={loss_main.item():.4f}, loss_aux={float(loss_aux):.4f}, loss_static={float(loss_static):.4f}")

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # METRICHE
        bs = rgb.size(0)
        total_loss += loss.item() * bs
        n_samples += bs    

        probs = torch.sigmoid(main_logits.detach().float())
        preds = (probs >= THR).float()

        if torch.isnan(probs).any():
            continue
        
        # skippa batch con mixup
        if use_mixup:
            continue
        else:
            target_metric = labels
        
        y_true.extend(target_metric.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_score.extend(probs.cpu().numpy())

        if (b_idx + 1) % log_interval == 0:
            iterator.set_postfix({
                "L_tot":   float(loss.item()),
                "L_main":  float(loss_main.item()),
                "L_stat":  float(loss_static.item()) if isinstance(loss_static, torch.Tensor) else float(loss_static),
                "L_aux":   float(loss_aux.item())    if isinstance(loss_aux, torch.Tensor) else float(loss_aux),
            }, refresh=False)

    metrics = compute_and_log_metrics(
        run_dir=run_dir_global,
        all_true=y_true,
        all_pred=y_pred,
        all_score=y_score,
        threshold=THR,
        class_names=("no_smoke", "smoke"),
        save_metrics=True,
        save_counts=False,
        verbose=True,
        dataset_label="train",
        current_epoch=ep,
        save_curves=True
    )
    metrics["loss"] = total_loss / max(1, n_samples)
    return metrics


# EVAL
@torch.no_grad()
def val_loop(model, loader, device, ce_loss_fn, ep=None, split_name="val"):
    global run_dir_global
    model.eval()

    total_loss = 0.0
    n_samples = 0
    y_true, y_score = [], []

    iterator = tqdm(enumerate(loader), total=len(loader),
                    desc=split_name, leave=True, dynamic_ncols=True)

    for b_idx, batch in iterator:
        rgb = batch["rgb"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).float()

        output = model(rgb)
        logits = output[0] if isinstance(output, tuple) else output
        
        weights = batch.get("weight", torch.ones_like(labels)).to(device)
        loss_raw = ce_loss_fn(logits, labels)
        loss = (loss_raw * weights).mean() 

        bs = rgb.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

        probs = torch.sigmoid(logits)

        y_true.extend(labels.detach().cpu().numpy().reshape(-1).tolist())
        y_score.extend(probs.detach().cpu().numpy().reshape(-1).tolist())

        if (b_idx + 1) % 20 == 0:
            iterator.set_postfix({"loss": f"{loss.item():.4f}"})

    y_true_np  = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_score_np = np.asarray(y_score, dtype=np.float32).reshape(-1)

    metrics = compute_and_log_metrics(
        run_dir=run_dir_global,
        all_true=y_true_np,
        all_pred=(y_score_np >= THR).astype(np.float32),
        all_score=y_score_np,
        threshold=THR,
        class_names=("no_smoke", "smoke"),
        save_metrics=True,
        save_counts=False,
        verbose=True,
        dataset_label=split_name,
        current_epoch=ep,
        save_curves=False
    )

    metrics["loss"] = total_loss / max(1, n_samples)
    return metrics


# MAIN
def main():
    global run_dir_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", type=str, default="/rise/deep-smoke-machine/back-end/data/split/metadata_train_split_2_by_camera.json")
    ap.add_argument("--val_json", type=str, default="/rise/deep-smoke-machine/back-end/data/split/metadata_validation_split_2_by_camera.json")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--num_workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--num_segments", type=int, default=DEFAULT_NUM_SEGMENTS)
    ap.add_argument("--rgb_dir", type=str, default=DEFAULT_RGB_DIR)
    ap.add_argument("--resume_checkpoint", type=str, default=None)
    ap.add_argument("--finetune", action="store_true", help="Usa LR ridotti per fine-tuning da pesi pre-addestrati")
    ap.add_argument("--run_dir", type=str, default=None, help="Directory della run (tutti i job a 1 epoca scrivono qui)")
    ap.add_argument("--one_epoch", type=int, default=None, help="Esegui solo N epoche (per debug/testing)")

    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.run_dir is not None:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
        run_name = os.path.basename(run_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"run-{timestamp}"
        run_dir = os.path.join(script_dir, "runs", run_name)
        os.makedirs(run_dir, exist_ok=True)

    run_dir_global = run_dir
    print(f"[RUN] Directory run: {run_dir}")

    train_transform = helper.get_transform(mode="rgb", phase="train", image_size=224)
    val_transform   = helper.get_transform(mode="rgb", phase="val",   image_size=224)
     
    model = MazingaSmokeClassifier(args.num_segments).to(device)

    # DEFINIZIONE PARAM GROUPS + OPTIMIZER
    # --- tau separati ---
    tau_params = [
        model.fusion_module.tau_spat,
        model.fusion_module.tau_gate,
        model.tau_static,
    ]
    tau_ids = {id(p) for p in tau_params}

    # --- head: tutto quello che non è backbone e non è tau ---
    head_params = (
        list(model.static_branch.parameters()) +
        list(model.temporal_branch.parameters()) +
        [p for p in model.fusion_module.parameters() if id(p) not in tau_ids] +
        list(model.mlp.parameters()) +
        list(model.aux_head.parameters()) +
        list(model.static_head.parameters()) +
        list(model.static_gate.parameters()) +
        list(model.static_in_proj.parameters()) +
        list(model.delta_proj.parameters()) +
        list(model.cepool.parameters())
    )

    base_optim = AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-4, "weight_decay": 5e-4},
        {"params": head_params,                "lr": 5e-4, "weight_decay": 5e-3},
        {"params": tau_params,                 "lr": 3e-5, "weight_decay": 0.0},
    ])

    optimizer = Lookahead(base_optim, k=5, alpha=0.5)

    # SCALING LR PER FINE-TUNING
    if args.finetune:
        apply_finetune_lr(optimizer)
        print("[FINETUNE] LR impostati manualmente (backbone molto basso).")
        DO_MIXUP = False
    else:
        DO_MIXUP = True
    
    warmup_epochs = 0 if args.finetune else 3

    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.3, total_iters=warmup_epochs)
    else:
        warmup = None

    base_min_lr = min(pg["lr"] for pg in optimizer.param_groups)
    eta_min = base_min_lr * 0.01

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - warmup_epochs),
        eta_min=eta_min
    )

    # Checkpoint init 
    checkpoint_path  = os.path.join(run_dir, "best_model.pt")
    last_ckpt_path   = os.path.join(run_dir, "checkpoint_full.pt")
    bestval_txt_path = os.path.join(run_dir, "best_val.txt")
    metrics_log_path = os.path.join(run_dir, "metrics_log.txt")

    best_f1_path     = os.path.join(run_dir, "best_f1.pt")
    best_auc_path = os.path.join(run_dir, "best_auc.pt")

    bestauc_txt_path = os.path.join(run_dir, "best_auc.txt")
    bestf1_txt_path  = os.path.join(run_dir, "best_f1.txt")

    start_epoch   = 1
    best_val_loss = float('inf')
    best_epoch    = None
    best_metrics  = None
    
    best_auc = float("-inf")
    best_auc_epoch = None
    best_auc_metrics = None

    best_f1 = float("-inf")
    best_f1_epoch = None
    best_f1_metrics = None

    # --- ricarica policy run ---
    meta = default_meta()

    policy = resolve_resume_policy(args, last_ckpt_path, FINETUNE_INIT_WEIGHTS)

    if policy["action"] == "resume_full":
        print(f"[RESUME FULL] Riprendo da {policy['ckpt_path']}")
        meta = load_full_checkpoint(
            policy["ckpt_path"],
            model,
            optimizer=optimizer,
            warmup=warmup,
            cosine=cosine,
            device=device
        )
        last_epoch = int(meta.get("epoch", 0))
        start_epoch = last_epoch + 1

    elif policy["action"] == "init_weights_only":
        print(f"[FINETUNE][INIT WEIGHTS ONLY] Carico SOLO pesi da {policy['ckpt_path']}")
        load_model_weights_only(policy["ckpt_path"], model, device)
        apply_finetune_freeze(model)
        print("[FINETUNE] Freeze conv_stem + blocks 0-1 della backbone.")
        start_epoch = 1

    else:
        start_epoch = 1

    # ripristina best dallo stato meta
    best_val_loss = meta.get("best_val_loss", best_val_loss)
    best_epoch = meta.get("best_epoch", best_epoch)
    best_metrics = meta.get("best_metrics", best_metrics)

    best_auc = meta.get("best_auc", best_auc)
    best_auc_epoch = meta.get("best_auc_epoch", best_auc_epoch)
    best_auc_metrics = meta.get("best_auc_metrics", best_auc_metrics)

    best_f1 = meta.get("best_f1", best_f1)
    best_f1_epoch = meta.get("best_f1_epoch", best_f1_epoch)
    best_f1_metrics = meta.get("best_f1_metrics", best_f1_metrics)


    if not os.path.isfile(metrics_log_path):
        with open(metrics_log_path, 'w') as lf:
            lf.write("# epoch\tset\tloss\tf1\tprecision\trecall\troc_auc\n")
            

    if args.one_epoch is not None:
        dl_epoch = int(args.one_epoch)
    else:
        dl_epoch = int(start_epoch)

    train_ds = loadDataset(args.train_json, args.rgb_dir, args.num_segments, transform=train_transform)
    val_ds = loadDataset(args.val_json,   args.rgb_dir, args.num_segments, transform=val_transform)

    train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
    prefetch_factor=4
)
    val_loader = DataLoader(
    val_ds,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True
)
    alpha_pos, alpha_neg = compute_focal_alphas(train_ds)

    #doppia focal loss
    ce_loss_clean = BinaryFocalLossBalanced(alpha_pos, alpha_neg, gamma=2.0, reduction='none').to(device)
    ce_loss_mix = BinaryFocalLossBalanced(1.0, 1.0, gamma=0.5, reduction='none').to(device)

        
    if args.one_epoch is not None:
        epochs_to_run = [args.one_epoch]
        #epochs_to_run = range(args.one_epoch, min(args.one_epoch + 3, args.epochs + 1))
    else:
        epochs_to_run = range(start_epoch, args.epochs + 1)

    for ep in epochs_to_run:
        print(f"\n===== EPOCH {ep}/{args.epochs} (best so far: {best_epoch if best_epoch is not None else '-'} | best_val_loss={best_val_loss:.4f}) =====")

        train_metrics = train_loop(model, train_loader, optimizer, device, ce_loss_clean, ce_loss_mix, DO_MIXUP, ep=ep)
        
        print(f"[TRAIN METRICS] loss={train_metrics['loss']:.4f} F1={train_metrics.get('f1_macro', float('nan')):.4f} "
              f"P={train_metrics.get('precision_macro', float('nan')):.4f} R={train_metrics.get('recall_macro', float('nan')):.4f} "
              f"AUC={train_metrics['roc_auc']:.4f}")
        with open(metrics_log_path, 'a') as lf:
            lf.write(f"{ep}\ttrain\t{train_metrics['loss']:.6f}\t{train_metrics['f1_macro']:.6f}\t{train_metrics['precision_macro']:.6f}\t{train_metrics['recall_macro']:.6f}\t{train_metrics['roc_auc']:.6f}\n")

        val_metrics = val_loop(model, val_loader, device, ce_loss_clean, ep=ep, split_name="val")


        print(f"[VAL METRICS] loss={val_metrics['loss']:.4f} F1={val_metrics['f1_macro']:.4f} P={val_metrics['precision_macro']:.4f} R={val_metrics['recall_macro']:.4f} AUC={val_metrics['roc_auc']:.4f}")
        with open(metrics_log_path, 'a') as lf:
            lf.write(f"{ep}\tval\t{val_metrics['loss']:.6f}\t{val_metrics['f1_macro']:.6f}\t{val_metrics['precision_macro']:.6f}\t{val_metrics['recall_macro']:.6f}\t{val_metrics['roc_auc']:.6f}\n")

        current_auc = float(val_metrics.get("roc_auc", float("nan")))
        current_f1  = float(val_metrics.get("f1_macro", float("nan")))   # F1 a THR=0.5

        # ---- BEST AUC ----
        if np.isfinite(current_auc) and current_auc > best_auc:
            best_auc = current_auc
            best_auc_epoch = ep
            best_auc_metrics = dict(val_metrics)
            torch.save(model.state_dict(), best_auc_path)
            with open(bestauc_txt_path, "w") as f:
                f.write(f"{best_auc:.8f}\n")
            print(f"[EPOCH {ep}] Nuovo BEST AUC={best_auc:.6f} -> {best_auc_path}")

        # ---- BEST F1  ----
        if np.isfinite(current_f1) and current_f1 > best_f1:
            best_f1 = current_f1
            best_f1_epoch = ep
            best_f1_metrics = dict(val_metrics)
            torch.save(model.state_dict(), best_f1_path)
            with open(bestf1_txt_path, "w") as f:
                f.write(f"{best_f1:.8f}\n")
            print(f"[EPOCH {ep}] Nuovo BEST F1@THR={best_f1:.6f} -> {best_f1_path}")

        if ep <= warmup_epochs:
            warmup.step()
        else:
            cosine.step()

        for i,pg in enumerate(optimizer.param_groups):
            print(f"[ep {ep}] LR[{i}] = {pg['lr']:.3e}")

        # checkpoint best
        current_val_loss = val_metrics['loss']
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_epoch    = ep
            best_metrics  = val_metrics
            torch.save(model.state_dict(), checkpoint_path)
            with open(bestval_txt_path, 'w') as f:
                f.write(f"{best_val_loss:.8f}\n")
            print(f"[EPOCH {ep}] Nuovo best VAL LOSS={best_val_loss:.4f} -> {checkpoint_path}")

        # checkpoint last
        try:
            meta.update({
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,

                "best_auc": best_auc,
                "best_auc_epoch": best_auc_epoch,
                "best_auc_metrics": best_auc_metrics,

                "best_f1": best_f1,
                "best_f1_epoch": best_f1_epoch,
                "best_f1_metrics": best_f1_metrics,
            })

            save_full_checkpoint(
                last_ckpt_path,
                epoch=ep,
                model=model,
                optimizer=optimizer,
                warmup=warmup,
                cosine=cosine,
                meta=meta
            )
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[LAST][WARN] Salvataggio last_checkpoint_full fallito: {e}")

    if best_metrics:
        print(f"\n===== BEST EPOCH SUMMARY =====")
        print(f"Best epoch: {best_epoch}  ValLoss={best_val_loss:.4f}")
        print(f"VAL METRICS: F1={best_metrics['f1_macro']:.4f} P={best_metrics['precision_macro']:.4f} R={best_metrics['recall_macro']:.4f} AUC={best_metrics['roc_auc']:.4f}")
    print(f"[TRAINING COMPLETATO] Best val_loss={best_val_loss:.4f} (file: {checkpoint_path})")

if __name__ == "__main__":
    main()