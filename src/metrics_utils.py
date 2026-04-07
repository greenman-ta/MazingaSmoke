"""
This file is obtained and modified from:
- https://github.com/CMU-CREATE-Lab/deep-smoke-machine
"""
import os
from typing import List, Tuple, Optional, Dict, Any
from sklearn.metrics import classification_report as cr
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, average_precision_score
import numpy as np
import matplotlib.pyplot as plt

def compute_and_log_metrics(
    run_dir: str,
    all_true: List[int],
    all_pred: List[int],
    all_score: List[float],
    threshold: float = 0.35,
    class_names: Tuple[str, str] = ("no_smoke", "smoke"),
    save_metrics: bool = True,
    save_counts: bool = True,
    verbose: bool = True,
    dataset_label: str = "set",
    save_curves: bool = True,
    current_epoch: Optional[int] = None,
) -> Dict[str, Any]:
    """Calcola metriche, salva file testuali e curve """

    results: Dict[str, Any] = {}
    if len(all_true) == 0:
        if verbose:
            print("[METRICS] Nessuna etichetta: salto calcolo metriche.")
        return results

    # Metriche principali  
    prec, rec, f1, _ = prfs(all_true, all_pred, average='binary', pos_label=1, zero_division=0)
    prec_macro, rec_macro, f1_macro, _ = prfs(all_true, all_pred, average='macro', zero_division=0)
    prec_w, rec_w, f1_w, _ = prfs(all_true, all_pred, average='weighted', zero_division=0)

    # ROC-AUC su score continui 
    try:
        roc_auc = roc_auc_score(all_true, all_score)
    except ValueError:
        roc_auc = float('nan')

    report = cr(all_true, all_pred, target_names=list(class_names), zero_division=0)

    # Conteggi 
    true_no = sum(1 for v in all_true if v == 0)
    true_yes = sum(1 for v in all_true if v == 1)
    pred_no = sum(1 for v in all_pred if v == 0)
    pred_yes = sum(1 for v in all_pred if v == 1)
    tp = sum(1 for t,p in zip(all_true, all_pred) if t==1 and p==1)
    tn = sum(1 for t,p in zip(all_true, all_pred) if t==0 and p==0)
    fp = sum(1 for t,p in zip(all_true, all_pred) if t==0 and p==1)
    fn = sum(1 for t,p in zip(all_true, all_pred) if t==1 and p==0)

    results.update({
        'precision_binary_smoke': prec,
        'recall_binary_smoke': rec,
        'f1_binary_smoke': f1,
        'roc_auc': roc_auc,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'f1_macro': f1_macro,
        'precision_weighted': prec_w,
        'recall_weighted': rec_w,
        'f1_weighted': f1_w,
        'true_no_smoke': true_no,
        'true_smoke': true_yes,
        'pred_no_smoke': pred_no,
        'pred_smoke': pred_yes,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'threshold': threshold,
        'classification_report': report,
    })

    if verbose:
        print(f"\n===== METRICHE [{dataset_label}] =====")
        print(f"Threshold decisione: {threshold}")
        print(f"Precision (binary smoke): {prec:.4f}")
        print(f"Recall    (binary smoke): {rec:.4f}")
        print(f"F1        (binary smoke): {f1:.4f}")
        print(f"ROC-AUC                 : {roc_auc:.4f}" if roc_auc==roc_auc else "ROC-AUC                 : NaN (una sola classe presente)")
        print("-- Macro avg --")
        print(f"Precision macro: {prec_macro:.4f}  Recall macro: {rec_macro:.4f}  F1 macro: {f1_macro:.4f}")
        print("-- Weighted avg --")
        print(f"Precision weighted: {prec_w:.4f}  Recall weighted: {rec_w:.4f}  F1 weighted: {f1_w:.4f}")
        print("-- Counts --")
        print(f"True no_smoke={true_no}  True smoke={true_yes}  Pred no_smoke={pred_no}  Pred smoke={pred_yes}")
        print(f"Confusion matrix (tn fp / fn tp): {tn} {fp} / {fn} {tp}")
        print("\nClassification report:\n" + report)

    # Salvataggi testuali 
    safe_label = dataset_label.lower().replace(' ','_')
    if save_metrics:
        metrics_path = os.path.join(run_dir, f'metrics_{safe_label}.txt')
        with open(metrics_path, 'a') as mf:
            mf.write(f"# Metriche {dataset_label}\n")
            for k in [
                'precision_binary_smoke','recall_binary_smoke','f1_binary_smoke','roc_auc',
                'precision_macro','recall_macro','f1_macro',
                'precision_weighted','recall_weighted','f1_weighted']:
                v = results[k]
                mf.write(f"{k}\t{v if v==v else 'NaN'}\n")
            mf.write("\nClassification report:\n")
            mf.write(report + "\n")

    if save_counts:
        counts_path = os.path.join(run_dir, f'counts_summary_{safe_label}.txt')
        with open(counts_path, 'w') as cf:
            cf.write(f"# Conteggi clip-level {dataset_label}\n")
            cf.write(f"threshold_smoke\t{threshold}\n")
            cf.write(f"true_no_smoke\t{true_no}\n")
            cf.write(f"true_smoke\t{true_yes}\n")
            cf.write(f"pred_no_smoke\t{pred_no}\n")
            cf.write(f"pred_smoke\t{pred_yes}\n")
            cf.write("# confusion_matrix tn fp fn tp\n")
            cf.write(f"{tn}\t{fp}\t{fn}\t{tp}\n")

    if save_curves and len(set(all_true)) >= 2:
        y_true_np  = np.asarray(all_true)
        y_score_np = np.asarray(all_score)

        # ROC
        fpr, tpr, thr = roc_curve(y_true_np, y_score_np)
        roc_auc_val   = auc(fpr, tpr)

        # PR
        prec_c, rec_c, thr_pr = precision_recall_curve(y_true_np, y_score_np)
        ap_val = average_precision_score(y_true_np, y_score_np)

        curves_dir = os.path.join(run_dir, "curves", safe_label)
        os.makedirs(curves_dir, exist_ok=True)
        tag = f"epoch_{int(current_epoch):02d}" if current_epoch is not None else "latest"

        # Dati grezzi
        np.savez(os.path.join(curves_dir, f"roc_{tag}.npz"),
                 fpr=fpr, tpr=tpr, thresholds=thr, auc=roc_auc_val)
        np.savez(os.path.join(curves_dir, f"pr_{tag}.npz"),
                 precision=prec_c, recall=rec_c, thresholds=thr_pr, ap=ap_val)

        # Figure ROC
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_val:.3f}")
        plt.plot([0,1],[0,1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC – {dataset_label} {tag}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(curves_dir, f"roc_{tag}.png"), dpi=150)
        plt.close()

        # Figure PR
        plt.figure()
        plt.plot(rec_c, prec_c, label=f"AP = {ap_val:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR – {dataset_label} {tag}")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(curves_dir, f"pr_{tag}.png"), dpi=150)
        plt.close()

        results["roc_auc_curve"] = float(roc_auc_val)
        results["average_precision"] = float(ap_val)

    return results