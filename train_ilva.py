import torch
from torch.optim import AdamW
from sys import path
import os, argparse
from torch.utils.data import DataLoader, Dataset
from torch import nn
from datetime import datetime
from tqdm import tqdm
import numpy as np
from timm.optim import Lookahead
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts,LinearLR, ReduceLROnPlateau
from torch.utils.data import WeightedRandomSampler

# MODEL 
from mazinga_smoke.mazinga_smoke_v2 import MazingaSmokeClassifier

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

from sklearn.metrics import precision_recall_curve
import json
from src.util_2 import bootstrap_targets, focal_gamma_schedule, set_loss_gamma

# CONFIG
DEFAULT_EPOCHS = 50
DEFAULT_LR = 5e-4
DEFAULT_BATCH = 18
DEFAULT_WORKERS = 4
DEFAULT_NUM_SEGMENTS = 36
DEFAULT_RGB_DIR = "/deep-smoke-machine/back-end/data/rgb224"
THR = 0.5

N_VIS = 40
MAX_PER_CLASS = 10

FINETUNE_INIT_WEIGHTS = "/mazingasmoke/pesi_v2/split_4.pt"

ILVA_RGB_DIR   = "/ilva/npy"
ILVA_TRAIN_JSON = "/ilva/split/metadata_ilva_train.json"
ILVA_VAL_JSON   = "/ilva/split/metadata_ilva_val.json"  
USE_ILVA_MIX = True
RISE_PER_REAL = 1 
run_dir_global = "."

def load_labels_and_weights_fast(json_path):
    """
    Estrae sia le label che i pesi manuali dal JSON
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for k in ["clips", "items", "data", "samples", "videos"]:
            if k in data and isinstance(data[k], list):
                data = data[k]
                break

    labels = []
    weights = []
    for item in data:
        labels.append(int(item.get("label", item.get("target", item.get("smoke", 0)))))
        # Recupera il peso manuale, default 1.0 se non specificato
        weights.append(float(item.get("weight", 1.0)))
    
    return labels, weights

def coral_loss(source, target):
    """
    Calcola la Correlation Alignment Loss tra due distribuzioni
    """
    d = source.size(1) if source.dim() > 1 else 1

    # covarianza source
    xm = source - source.mean(dim=0, keepdim=True)
    xc = (xm.t() @ xm) / (source.size(0) - 1 + 1e-8)
    
    # covarianza target
    tm = target - target.mean(dim=0, keepdim=True)
    tc = (tm.t() @ tm) / (target.size(0) - 1 + 1e-8)
    
    # Distanza di Frobenius tra le covarianze
    loss = torch.sum((xc - tc) ** 2) / (4 * (d ** 2))
    return loss

def best_f1_threshold(y_true, y_score):
    """
    Ritorna: thr_best, f1_best, p_best, r_best
    """
    y_true  = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float32).reshape(-1)

    if len(np.unique(y_true)) < 2:
        return 0.5, float("nan"), float("nan"), float("nan")

    p, r, thr = precision_recall_curve(y_true, y_score)  # thr len = len(p)-1
    if thr.size == 0:
        return 0.5, float("nan"), float("nan"), float("nan")

    f1 = 2 * p * r / (p + r + 1e-12)

    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i]), float(f1[i]), float(p[i]), float(r[i])

def strip_raw_metrics(m):
    m = dict(m)
    m.pop("_y_true", None)
    m.pop("_y_score", None)
    return m

class HybridBatchLoader:
    """
    batch ibrido: concat(Source=RISE, Target=ILVA)
    l'epoca è definita da ILVA
    """
    def __init__(self, loader_source, loader_target):
        self.loader_source = loader_source
        self.loader_target = loader_target
        self.length = len(loader_target)  

    def __len__(self):
        return self.length

    def __iter__(self):
        it_source = iter(self.loader_source)

        for batch_t in self.loader_target:  # <-- quando ILVA finisce, STOP
            try:
                batch_s = next(it_source)
            except StopIteration:
                it_source = iter(self.loader_source)
                batch_s = next(it_source)

            hybrid_batch = {}
            for k in batch_s.keys():
                if isinstance(batch_s[k], torch.Tensor):
                    hybrid_batch[k] = torch.cat([batch_s[k], batch_t[k]], dim=0)
                elif isinstance(batch_s[k], list):
                    hybrid_batch[k] = batch_s[k] + batch_t[k]

            domain_labels = torch.cat([
                torch.zeros(batch_s['rgb'].size(0)),
                torch.ones(batch_t['rgb'].size(0))
            ], dim=0)
            hybrid_batch["domain"] = domain_labels

            yield hybrid_batch
            
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

def get_lambda_domain(ep: int) -> float:
    if ep <= 2:
        return 5.0
    elif ep <= 4:
        return 15.0
    else:
        return 35.0
    
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

    coral_sum = 0.0
    coral_sum_weighted = 0.0
    coral_count = 0

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
    
    # ---- FOCAL ANNEALED ----
    g_clean = focal_gamma_schedule(ep, warm_end=1, ramp_end=8, g0=0.0, g_warm=0.3, g_final=1.0)
    set_loss_gamma(ce_loss_fn, g_clean)
    set_loss_gamma(ce_loss_mix, 0.5)



    for b_idx, batch in iterator:

        rgb = batch["rgb"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).float()
        domain = batch.get("domain", torch.zeros_like(labels)).to(device, non_blocking=True)
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

        output = model(input_var, ft=True)  # ft=True per ottenere anche le features latenti per CORAL

        # Gestione output

        if isinstance(output, tuple):
            if len(output) == 3:
                main_logits, aux_logits, static_logits = output
            elif len(output) == 4:
                main_logits, aux_logits, static_logits, latent_features = output
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
        loss_raw  = loss_fn(main_logits, labels_mix if use_mixup else labels)
        loss_main = (loss_raw * weight_mixed).mean()


        # LOSS AUSILIARIA (TEMPORALE)
        loss_aux = 0.0
        if aux_logits is not None:
            loss_aux_raw = loss_fn(aux_logits, labels_mix if use_mixup else labels)
            loss_aux = (loss_aux_raw * weight_mixed).mean()

        # STATIC
        loss_static = 0.0
        if static_logits is not None:
            loss_static_raw = loss_fn(static_logits, labels_mix if use_mixup else labels)
            loss_static = (loss_static_raw * weight_mixed).mean()

        # DOMAIN ALIGNMENT (CORAL LOSS) 
        loss_domain = 0.0
        lambda_domain = get_lambda_domain(ep)  

        domain = batch.get("domain", torch.zeros_like(labels)).to(device, non_blocking=True).long()

        ms = (domain == 0)
        mt = (domain == 1)

        # allinea per classe, ma solo se hai abbastanza esempi
        min_n = 2 

        if latent_features is not None and ms.any() and mt.any():

            # labels è float, rendiamolo binario {0,1} pulito
            y = (labels > 0.5)

            # CORAL per classe 0 e 1
            for c in [False, True]:
                ms_c = ms & (y == c)
                mt_c = mt & (y == c)

                ns = int(ms_c.sum().item())
                nt = int(mt_c.sum().item())

                if ns >= min_n and nt >= min_n:
                    loss_domain = loss_domain + coral_loss(latent_features[ms_c], latent_features[mt_c])

            # --- LOG CORAL ---
            
            if isinstance(loss_domain, torch.Tensor):
                coral_val = float(loss_domain.detach().cpu().item())
            else:
                coral_val = float(loss_domain)

            coral_sum += coral_val
            coral_sum_weighted += (lambda_domain * coral_val)
            coral_count += 1

            if (b_idx + 1) % 10 == 0:
                    print(
                        f"[ep {ep} b{b_idx+1}] CORAL raw={coral_val:.6f} "
                        f"lambda={lambda_domain:.3f} weighted={(lambda_domain*coral_val):.6f}",
                        flush=True
                    )
                

        # LOSS TOTALE
        loss = loss_main + loss_aux * 1.0 + loss_static * 1.0 + loss_domain * lambda_domain


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
        
        # per le metriche: con same-class mixup label_a e label_b sono della stessa classe,
        # quindi scegliere label_a o label_b non cambia la semantica
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

    if coral_count > 0:
        coral_avg = coral_sum / coral_count
        coral_avg_w = coral_sum_weighted / coral_count
        print(f"[ep {ep}] CORAL avg raw={coral_avg:.6f} | avg weighted={coral_avg_w:.6f}")
        metrics["coral_raw_avg"] = coral_avg
        metrics["coral_weighted_avg"] = coral_avg_w

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

    # aggiungo raw per best-thr, debug
    metrics["_y_true"] = y_true_np
    metrics["_y_score"] = y_score_np

    return metrics

# MAIN
def main():
    global run_dir_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", type=str, default="/rise/deep-smoke-machine/back-end/data/split/metadata_train_split_by_date.json")
    ap.add_argument("--val_json", type=str, default="/rise/deep-smoke-machine/back-end/data/split/metadata_validation_split_by_date.json")
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

    # Checkpoint init 
    checkpoint_path  = os.path.join(run_dir, "best_model.pt")
    last_ckpt_path   = os.path.join(run_dir, "checkpoint_full.pt")
    bestval_txt_path = os.path.join(run_dir, "best_val.txt")
    metrics_log_path = os.path.join(run_dir, "metrics_log.txt")

    best_f1_path     = os.path.join(run_dir, "best_f1.pt")
    best_auc_path = os.path.join(run_dir, "best_auc.pt")

    bestauc_txt_path = os.path.join(run_dir, "best_auc.txt")
    bestf1_txt_path  = os.path.join(run_dir, "best_f1.txt")

    # ricarica policy run
    meta = default_meta()
    policy = resolve_resume_policy(args, last_ckpt_path, FINETUNE_INIT_WEIGHTS)

    model = MazingaSmokeClassifier(args.num_segments).to(device)
    start_epoch = 1

    if policy["action"] == "init_weights_only":
        print(f"[FINETUNE][INIT WEIGHTS ONLY] Carico SOLO pesi da {policy['ckpt_path']}")
        load_model_weights_only(policy["ckpt_path"], model, device)
        start_epoch = 1
    if args.finetune:
        apply_finetune_freeze(model)
        print("[FINETUNE] Freeze conv_stem + blocks 0-1 della backbone.")

    half_batch = max(1, args.batch_size // 2)

    train_transform = helper.get_transform(mode="rgb", phase="train", image_size=224)
    val_transform   = helper.get_transform(mode="rgb", phase="val",   image_size=224)

    # RISE 
    train_ds_rise = loadDataset(args.train_json, args.rgb_dir, args.num_segments, transform=train_transform)
    val_ds_rise   = loadDataset(args.val_json,   args.rgb_dir, args.num_segments, transform=val_transform)

    train_loader_rise = DataLoader(
        train_ds_rise,
        batch_size=half_batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    val_loader_rise = DataLoader(
        val_ds_rise,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # ILVA
    if USE_ILVA_MIX and args.finetune:
        train_ds_ilva = loadDataset(ILVA_TRAIN_JSON, ILVA_RGB_DIR, args.num_segments, transform=train_transform)
        val_ds_ilva   = loadDataset(ILVA_VAL_JSON,   ILVA_RGB_DIR, args.num_segments, transform=val_transform)

        val_loader_ilva = DataLoader(
            val_ds_ilva,
            batch_size=max(1, args.batch_size // 2),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )        

        # Dimezziamo il batch size per i singoli loader
        labels_ilva, base_weights_ilva = load_labels_and_weights_fast(ILVA_TRAIN_JSON)
        # sanity check
        if len(labels_ilva) != len(train_ds_ilva):
            raise RuntimeError(f"Mismatch: labels={len(labels_ilva)} vs dataset={len(train_ds_ilva)}. "
                            f"Probabile ordine/filtri diversi tra JSON e dataset.")

        # Ricalcolo dei pesi per mantenere il 20% garantendo i pesi manuali
        p_target = 0.20
        sum_w_neg = sum(w for l, w in zip(labels_ilva, base_weights_ilva) if l == 0)
        sum_w_pos = sum(w for l, w in zip(labels_ilva, base_weights_ilva) if l == 1)
        
        # Moltiplicatore esatto per far pesare il totale della classe positiva come p_target del totale pesato
        w_pos_mult = (p_target / (1.0 - p_target)) * (sum_w_neg / max(1.0, sum_w_pos))
        w_pos_mult = float(min(w_pos_mult, 4.0))
        
        # Applichiamo il moltiplicatore solo alla classe positiva
        weights_ilva = [
            (w * w_pos_mult) if l == 1 else w
            for l, w in zip(labels_ilva, base_weights_ilva)
        ]

        sampler_ilva = WeightedRandomSampler(weights_ilva, num_samples=len(weights_ilva), replacement=True)

        train_loader_ilva = DataLoader(
            train_ds_ilva,
            batch_size=half_batch, 
            sampler=sampler_ilva,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4
        )

        train_loader_rise = DataLoader(
            train_ds_rise,
            batch_size=half_batch, 
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4
        )

        train_loader = HybridBatchLoader(train_loader_rise, train_loader_ilva)

        val_loader = val_loader_ilva
        val_name = "val_ilva"

        print(f"[MIX] finetune mix attivo: RISE:ILVA={RISE_PER_REAL}:1 "
            f"| steps/epoch={len(train_loader)} | len(ilva)={len(train_loader_ilva)}")

    else:
        train_loader = train_loader_rise
        val_loader   = val_loader_rise
        val_name = "val"

    if USE_ILVA_MIX and args.finetune:
        alpha_pos, alpha_neg = compute_focal_alphas(train_ds_ilva)
    else:
        alpha_pos, alpha_neg = compute_focal_alphas(train_ds_rise)

    # clamp
    alpha_pos = min(alpha_pos, 10.0)
    alpha_neg = min(alpha_neg, 10.0)

    # DEFINIZIONE PARAM GROUPS + OPTIMIZER

    # --- tau separati ---
    tau_params = [
        model.fusion_module.tau_spat,
        model.fusion_module.tau_gate,
        model.tau_static,
    ]
    tau_ids = {id(p) for p in tau_params}

    model_params = (
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

    backbone_trainable = [p for p in model.backbone.parameters() if p.requires_grad]


    base_optim = AdamW([
        {"params": backbone_trainable, "lr": 1e-4, "weight_decay": 5e-4},
        {"params": model_params,                "lr": 5e-4, "weight_decay": 5e-3},
        {"params": tau_params,                 "lr": 3e-5, "weight_decay": 0.0},
    ])

    optimizer = Lookahead(base_optim, k=5, alpha=0.5)

    print("[INIT] LRs:", [pg["lr"] for pg in optimizer.param_groups])

    opt_ids = set()
    for pg in optimizer.param_groups:
        for p in pg["params"]:
            opt_ids.add(id(p))

    missing = []
    for name, p in model.named_parameters():
        if p.requires_grad and (id(p) not in opt_ids):
            missing.append(name)

    print("Missing trainable params:", missing)
    print("Count missing:", len(missing))

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

    base_min_lr = min(g["lr"] for g in optimizer.param_groups)
    eta_min = base_min_lr * 0.01

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - warmup_epochs),
        eta_min=eta_min
    )
    print("[INIT] eta_min:", cosine.eta_min, "T_max:", cosine.T_max)


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


    # ripristina best dallo stato meta
    best_val_loss = meta.get("best_val_loss", best_val_loss)
    best_epoch    = meta.get("best_epoch", best_epoch)
    best_metrics  = meta.get("best_metrics", best_metrics)

    best_auc         = meta.get("best_auc", best_auc)
    best_auc_epoch   = meta.get("best_auc_epoch", best_auc_epoch)
    best_auc_metrics = meta.get("best_auc_metrics", best_auc_metrics)

    best_f1         = meta.get("best_f1", best_f1)
    best_f1_epoch   = meta.get("best_f1_epoch", best_f1_epoch)
    best_f1_metrics = meta.get("best_f1_metrics", best_f1_metrics)


    if not os.path.isfile(metrics_log_path):
        with open(metrics_log_path, 'w') as lf:
            lf.write("# epoch\tset\tloss\tf1\tprecision\trecall\troc_auc\n")
    alpha_pos, alpha_neg = 1.0, 1.0
    #doppia focal loss
    ce_loss_clean = BinaryFocalLossBalanced(alpha_pos, alpha_neg, gamma=2.0, reduction='none').to(device)
    ce_loss_mix   = BinaryFocalLossBalanced(1.0, 1.0, gamma=0.5, reduction='none').to(device)
        
    if args.one_epoch is not None:
        epochs_to_run = [args.one_epoch]
        #epochs_to_run = range(args.one_epoch, min(args.one_epoch + 10, args.epochs + 1))
    else:
        epochs_to_run = range(start_epoch, args.epochs + 1)

    for ep in epochs_to_run:
        print(f"\n===== EPOCH {ep}/{args.epochs} (best so far: {best_epoch if best_epoch is not None else '-'} | best_val_loss={best_val_loss:.4f}) =====")

        active_train_loader = train_loader

        train_metrics = train_loop(
            model,
            active_train_loader,
            optimizer,
            device,
            ce_loss_clean,
            ce_loss_mix,
            DO_MIXUP,
            ep=ep
        )

        print(f"[TRAIN METRICS] loss={train_metrics['loss']:.4f} F1={train_metrics.get('f1_macro', float('nan')):.4f} "
              f"P={train_metrics.get('precision_macro', float('nan')):.4f} R={train_metrics.get('recall_macro', float('nan')):.4f} "
              f"AUC={train_metrics['roc_auc']:.4f}")
        with open(metrics_log_path, 'a') as lf:
            lf.write(f"{ep}\ttrain\t{train_metrics['loss']:.6f}\t{train_metrics['f1_macro']:.6f}\t{train_metrics['precision_macro']:.6f}\t{train_metrics['recall_macro']:.6f}\t{train_metrics['roc_auc']:.6f}\n")

        val_metrics = val_loop(model, val_loader, device, ce_loss_clean, ep=ep, split_name=val_name)

        print(f"[VAL METRICS] loss={val_metrics['loss']:.4f} F1={val_metrics['f1_macro']:.4f} P={val_metrics['precision_macro']:.4f} R={val_metrics['recall_macro']:.4f} AUC={val_metrics['roc_auc']:.4f}")
        with open(metrics_log_path, 'a') as lf:
            lf.write(f"{ep}\tval\t{val_metrics['loss']:.6f}\t{val_metrics['f1_macro']:.6f}\t{val_metrics['precision_macro']:.6f}\t{val_metrics['recall_macro']:.6f}\t{val_metrics['roc_auc']:.6f}\n")

        if val_name == "val_ilva":
            y_true_ilva  = val_metrics.get("_y_true", None)
            y_score_ilva = val_metrics.get("_y_score", None)

            if y_true_ilva is not None and y_score_ilva is not None:
                thr_star, f1_star, p_star, r_star = best_f1_threshold(y_true_ilva, y_score_ilva)
                print(f"[ILVA][BEST_THR] thr*={thr_star:.3f}  F1*={f1_star:.4f}  P*={p_star:.4f}  R*={r_star:.4f}")

                # salva su file txt
                with open(os.path.join(run_dir_global, "best_thr_ilva.txt"), "a") as f:
                    f.write(f"{ep}\t{thr_star:.6f}\t{f1_star:.6f}\t{p_star:.6f}\t{r_star:.6f}\n")
                    
        current_auc = float(val_metrics.get("roc_auc", float("nan")))
        current_f1  = float(val_metrics.get("f1_macro", float("nan")))   # F1 a THR=0.5

        # BEST AUC 
        if np.isfinite(current_auc) and current_auc > best_auc:
            best_auc = current_auc
            best_auc_epoch = ep
            best_auc_metrics = strip_raw_metrics(val_metrics)
            torch.save(model.state_dict(), best_auc_path)
            with open(bestauc_txt_path, "w") as f:
                f.write(f"{best_auc:.8f}\n")
            print(f"[EPOCH {ep}] Nuovo BEST AUC={best_auc:.6f} -> {best_auc_path}")

        # BEST F1  
        if np.isfinite(current_f1) and current_f1 > best_f1:
            best_f1 = current_f1
            best_f1_epoch = ep
            best_f1_metrics  = strip_raw_metrics(val_metrics)
            torch.save(model.state_dict(), best_f1_path)
            with open(bestf1_txt_path, "w") as f:
                f.write(f"{best_f1:.8f}\n")
            print(f"[EPOCH {ep}] Nuovo BEST F1@THR={best_f1:.6f} -> {best_f1_path}")

        prev = [pg["lr"] for pg in optimizer.param_groups]

        if ep <= warmup_epochs:
            warmup.step()
        else:
            cosine.step()

        now = [pg["lr"] for pg in optimizer.param_groups]
        print(f"[ep {ep}] LR changed? {any(a!=b for a,b in zip(prev, now))} | prev={prev} | now={now}")
        for i,g in enumerate(optimizer.param_groups):
            print(f"[ep {ep}] LR[{i}] = {g['lr']:.3e}")

        # checkpoint best
        current_val_loss = val_metrics['loss']
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_epoch    = ep
            best_metrics      = strip_raw_metrics(val_metrics)
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