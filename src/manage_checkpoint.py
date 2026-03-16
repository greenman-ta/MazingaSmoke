import os
import torch


def default_meta():
    return {
        "best_val_loss": float("inf"),
        "best_epoch": None,
        "best_metrics": None,

        "best_auc": float("-inf"),
        "best_auc_epoch": None,
        "best_auc_metrics": None,

        "best_f1": float("-inf"),
        "best_f1_epoch": None,
        "best_f1_metrics": None,

        "best_f1c": float("-inf"),
        "best_f1c_epoch": None,
        "best_f1c_metrics": None,
    }


def load_model_weights_only(path, model, device="cpu"):
    ck = torch.load(path, map_location=device)
    state = ck["model_state"] if (isinstance(ck, dict) and "model_state" in ck) else ck
    try:
        model.load_state_dict(state)
    except RuntimeError:
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)


def load_full_checkpoint(path, model, optimizer=None, warmup=None, cosine=None, device="cpu"):
    ck = torch.load(path, map_location=device)
    meta = default_meta()

    if isinstance(ck, dict) and "model_state" in ck:
        # Model
        try:
            model.load_state_dict(ck["model_state"])
        except RuntimeError:
            new_state = {k.replace("module.", ""): v for k, v in ck["model_state"].items()}
            model.load_state_dict(new_state)

        # Optimizer
        if optimizer is not None and "optimizer_state" in ck:
            try:
                optimizer.load_state_dict(ck["optimizer_state"])
            except Exception:
                print("[CHECKPOINT][WARN] Impossibile ripristinare optimizer_state")

        # Schedulers
        if warmup is not None and "warmup_state" in ck and ck["warmup_state"] is not None:
            try:
                warmup.load_state_dict(ck["warmup_state"])
            except Exception:
                print("[CHECKPOINT][WARN] Impossibile ripristinare warmup_state")

        if cosine is not None and "cosine_state" in ck:
            try:
                cosine.load_state_dict(ck["cosine_state"])
            except Exception:
                print("[CHECKPOINT][WARN] Impossibile ripristinare cosine_state")

        # Meta
        for k in meta.keys():
            if k in ck:
                meta[k] = ck[k]

    else:
        # weights-only checkpoint
        try:
            model.load_state_dict(ck)
        except RuntimeError:
            new_state = {k.replace("module.", ""): v for k, v in ck.items()}
            model.load_state_dict(new_state)

    # Fallback: best_val.txt se presente
    try:
        ck_dir = os.path.dirname(os.path.abspath(path))
        bestval_path = os.path.join(ck_dir, "best_val.txt")
        if os.path.isfile(bestval_path):
            with open(bestval_path, "r") as bf:
                txt = bf.read().strip().splitlines()[0]
                meta["best_val_loss"] = float(txt.strip())
    except Exception:
        pass
    
    meta["epoch"] = int(ck["epoch"]) if (isinstance(ck, dict) and "epoch" in ck) else 0
    return meta


def save_full_checkpoint(path, epoch, model, optimizer, warmup, cosine, meta: dict):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "warmup_state": warmup.state_dict() if warmup is not None else None,
        "cosine_state": cosine.state_dict() if cosine is not None else None,
    }
    # merge meta
    if meta is not None:
        ckpt.update(meta)
    
    ckpt["epoch"] = epoch
    torch.save(ckpt, path)