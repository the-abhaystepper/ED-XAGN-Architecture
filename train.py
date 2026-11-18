import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data.dataset import CLASS_NAMES, get_dataloaders
from src.models.ed_xagn import EDXAGN

EPOCHS = 30
PATIENCE = 5
LAMBDA_AUX = 0.3


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, reduction="none")
    probs = F.softmax(logits, dim=1)
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    alpha_t = alpha.to(device=logits.device, dtype=logits.dtype)[targets]
    return (alpha_t * (1 - pt).pow(gamma) * ce).mean()


def train_one_epoch(model, loader, optimizer, device, class_alpha: torch.Tensor,
                    epoch: int, total_epochs: int, lambda_aux: float = LAMBDA_AUX):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", unit="batch", dynamic_ncols=True)
    for batch_idx, (images, labels, _pids) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        out = model(images, compute_uncertainty=False, return_heatmap=False)
        cls_logits = out["cls_logits"]
        aux_logits = out["aux_logits"]

        loss_cls = focal_loss(cls_logits, labels, class_alpha)
        loss_aux = focal_loss(aux_logits, labels, class_alpha)
        loss = loss_cls + lambda_aux * loss_aux

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds = cls_logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

        pbar.set_postfix({
            "loss": f"{total_loss / (batch_idx + 1):.4f}",
            "acc": f"{correct / total:.4f}",
        })

    return total_loss / len(loader), correct / total


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    idx = y_true * num_classes + y_pred
    cm = np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion": cm,
    }


def aggregate_patients(pids: list, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Score each patient once from that patient's slice probabilities.

    Mean uses the average softmax. Max uses the per-class maximum softmax.
    """
    grouped = {}
    for i, pid in enumerate(pids):
        grouped.setdefault(pid, []).append(i)

    labels = []
    mean_probs = []
    max_probs = []
    for idxs in grouped.values():
        idxs = np.asarray(idxs)
        patient_labels = y_true[idxs]
        if not np.all(patient_labels == patient_labels[0]):
            raise ValueError("A patient batch contains more than one label")
        probs = y_prob[idxs]
        labels.append(int(patient_labels[0]))
        mean_probs.append(probs.mean(axis=0))
        max_probs.append(probs.max(axis=0))

    labels = np.asarray(labels, dtype=np.int64)
    mean_probs = np.stack(mean_probs)
    max_probs = np.stack(max_probs)
    num_classes = y_prob.shape[1]
    return {
        "y_true": labels,
        "mean_prob": mean_probs,
        "mean_pred": mean_probs.argmax(axis=1),
        "mean_metrics": classification_metrics(labels, mean_probs.argmax(axis=1), num_classes),
        "max_prob": max_probs,
        "max_pred": max_probs.argmax(axis=1),
        "max_metrics": classification_metrics(labels, max_probs.argmax(axis=1), num_classes),
    }


def binary_roc(y_true: np.ndarray, scores: np.ndarray):
    y_true = y_true.astype(np.int32)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5

    order = np.argsort(scores, kind="mergesort")[::-1]
    y_sorted = y_true[order]
    scores_sorted = scores[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    distinct = np.r_[True, scores_sorted[1:] != scores_sorted[:-1]]
    tps = tps[distinct]
    fps = fps[distinct]
    fpr = np.r_[0.0, fps / n_neg]
    tpr = np.r_[0.0, tps / n_pos]
    if fpr[-1] < 1.0:
        fpr = np.r_[fpr, 1.0]
        tpr = np.r_[tpr, 1.0]
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc = float(trapz(tpr, fpr))
    return fpr, tpr, auc


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, class_names: list, out_path: Path, title: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    aucs = {}
    mean_fpr = np.linspace(0.0, 1.0, 100)
    mean_tpr = np.zeros_like(mean_fpr)

    for k, name in enumerate(class_names):
        fpr, tpr, auc = binary_roc((y_true == k).astype(np.int32), y_prob[:, k])
        aucs[name] = auc
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        ax.plot(fpr, tpr, lw=1.8, label=f"{name}  (AUC={auc:.3f})")

    mean_tpr /= len(class_names)
    macro_auc = float(np.mean(list(aucs.values())))
    ax.plot(mean_fpr, mean_tpr, color="black", lw=2.0, linestyle="--",
            label=f"macro-average  (AUC={macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="0.7", lw=1, linestyle=":")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    aucs["macro"] = macro_auc
    return aucs


def print_metrics(metrics: dict, class_names: list, prefix: str = "") -> None:
    print(f"{prefix}Precision / Recall / F1")
    for i, name in enumerate(class_names):
        print(f"{prefix}  {name:12s}  P={metrics['precision'][i]:.4f}  "
              f"R={metrics['recall'][i]:.4f}  F1={metrics['f1'][i]:.4f}")
    print(f"{prefix}  {'macro':12s}  P={metrics['macro_precision']:.4f}  "
          f"R={metrics['macro_recall']:.4f}  F1={metrics['macro_f1']:.4f}")


def print_aucs(aucs: dict, class_names: list, prefix: str = "") -> None:
    print(f"{prefix}ROC AUC (one-vs-rest)")
    for name in class_names:
        print(f"{prefix}  {name:12s}  AUC={aucs[name]:.4f}")
    print(f"{prefix}  {'macro':12s}  AUC={aucs['macro']:.4f}")


@torch.no_grad()
def evaluate(model, loader, device, class_alpha: torch.Tensor, epoch: int, total_epochs: int,
             split_name: str, lambda_aux: float = LAMBDA_AUX):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []
    all_pids = []

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs} [{split_name}]".ljust(28),
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )
    for batch_idx, (images, labels, pids) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        out = model(images, compute_uncertainty=False, return_heatmap=False)
        cls_logits = out["cls_logits"]
        aux_logits = out["aux_logits"]

        loss_cls = focal_loss(cls_logits, labels, class_alpha)
        loss_aux = focal_loss(aux_logits, labels, class_alpha)
        loss = loss_cls + lambda_aux * loss_aux

        probs = out["cls_probs"]
        preds = cls_logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())
        all_pids.extend(pids)

        pbar.set_postfix({
            "loss": f"{total_loss / (batch_idx + 1):.4f}",
            "acc": f"{correct / total:.4f}",
        })

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()
    num_classes = len(CLASS_NAMES)
    return {
        "loss": total_loss / len(loader),
        "slice_acc": correct / total,
        "slice_metrics": classification_metrics(y_true, y_pred, num_classes),
        "y_true": y_true,
        "y_prob": y_prob,
        "patients": aggregate_patients(all_pids, y_true, y_prob),
    }


def report_split(result: dict, title: str) -> None:
    patients = result["patients"]
    print(f"{title}")
    print(f"  slice accuracy: {result['slice_acc']:.4f}  loss: {result['loss']:.4f}")
    print_metrics(result["slice_metrics"], CLASS_NAMES, prefix="  ")
    print(f"  patient accuracy (mean pool): {(patients['mean_pred'] == patients['y_true']).mean():.4f}  "
          f"n={len(patients['y_true'])}")
    print_metrics(patients["mean_metrics"], CLASS_NAMES, prefix="  ")
    print(f"  patient accuracy (max pool): {(patients['max_pred'] == patients['y_true']).mean():.4f}")
    print_metrics(patients["max_metrics"], CLASS_NAMES, prefix="  ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train single-stream ED-XAGN on patient-grouped Figshare folds.")
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help=f"Run all {EPOCHS} epochs. The best validation patient macro-F1 checkpoint is still restored for the test set.",
    )
    parser.add_argument(
        "--roc-only",
        action="store_true",
        help="Skip training. Load edxagn.pt and write the validation patient-mean ROC.",
    )
    return parser.parse_args()


def write_roc(result: dict, out_path: Path, title: str) -> tuple[dict, dict]:
    patient_aucs = plot_roc(
        result["patients"]["y_true"],
        result["patients"]["mean_prob"],
        CLASS_NAMES,
        out_path,
        title,
    )
    slice_aucs = {}
    for k, name in enumerate(CLASS_NAMES):
        _, _, auc = binary_roc((result["y_true"] == k).astype(np.int32), result["y_prob"][:, k])
        slice_aucs[name] = auc
    slice_aucs["macro"] = float(np.mean([slice_aucs[name] for name in CLASS_NAMES]))
    return patient_aucs, slice_aucs


def roc_from_checkpoint(device: torch.device) -> None:
    ckpt_path = Path(__file__).resolve().parent / "edxagn.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    _, val_loader, _, class_alpha = get_dataloaders()
    model = EDXAGN(num_classes=len(CLASS_NAMES))
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    print(f"Loaded weights from {ckpt_path}")

    val = evaluate(model, val_loader, device, class_alpha, 0, 0, "Val", LAMBDA_AUX)
    report_split(val, "\nValidation (checkpoint)")
    roc_path = Path(__file__).resolve().parent / "roc_curve.png"
    patient_aucs, slice_aucs = write_roc(
        val,
        roc_path,
        "ED-XAGN one-vs-rest ROC (validation, patient mean)",
    )
    print("  Validation patient ROC (mean pool)")
    print_aucs(patient_aucs, CLASS_NAMES, prefix="  ")
    print("  Validation slice ROC")
    print_aucs(slice_aucs, CLASS_NAMES, prefix="  ")
    print(f"  ROC curve saved to {roc_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    if args.roc_only:
        roc_from_checkpoint(device)
        return
    if args.no_early_stopping:
        print(f"Early stopping off. Training for all {EPOCHS} epochs.")
    else:
        print(f"Early stopping on. Patience {PATIENCE} epochs of validation patient macro-F1.")

    train_loader, val_loader, test_loader, class_alpha = get_dataloaders()
    print(f"Training samples: {len(train_loader.dataset)} ({len(train_loader)} batches)")
    print(f"Validation samples: {len(val_loader.dataset)} ({len(val_loader)} batches)")
    print(f"Test samples: {len(test_loader.dataset)} ({len(test_loader)} batches)")

    print("Building single-stream ED-XAGN...", flush=True)
    model = EDXAGN(num_classes=len(CLASS_NAMES))
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"Moving model to {device}...", flush=True)
    model = model.to(device)
    for param in model.uncert_head.parameters():
        param.requires_grad = False
    print("Model ready.", flush=True)

    optimizer = Adam((param for param in model.parameters() if param.requires_grad), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    stall = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, class_alpha, epoch, EPOCHS, LAMBDA_AUX
        )
        val = evaluate(model, val_loader, device, class_alpha, epoch, EPOCHS, "Val", LAMBDA_AUX)
        scheduler.step()
        patient_f1 = val["patients"]["mean_metrics"]["macro_f1"]
        print(
            f"Epoch {epoch}/{EPOCHS} complete | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val['loss']:.4f} | Val slice acc: {val['slice_acc']:.4f} | "
            f"Val patient F1: {patient_f1:.4f} | Time: {time.time() - t0:.1f}s"
        )
        report_split(val, "  Validation")

        if patient_f1 > best_f1:
            best_f1 = patient_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_state["uncert_head.weight"] = best_state["class_head.weight"].clone()
            best_state["uncert_head.bias"] = best_state["class_head.bias"].clone()
            stall = 0
            print(f"  New best patient macro-F1 {best_f1:.4f} at epoch {best_epoch}")
        else:
            stall += 1
            print(f"  No improvement for {stall} epoch(s) (best {best_f1:.4f} at epoch {best_epoch})")
            if not args.no_early_stopping and stall >= PATIENCE:
                print(f"Early stop at epoch {epoch}. Restoring epoch {best_epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")

    model.load_state_dict(best_state)
    val = evaluate(model, val_loader, device, class_alpha, best_epoch, EPOCHS, "Val", LAMBDA_AUX)
    test = evaluate(model, test_loader, device, class_alpha, best_epoch, EPOCHS, "Test", LAMBDA_AUX)
    report_split(val, f"\nValidation (weights from epoch {best_epoch})")
    report_split(test, f"\nTest (weights from epoch {best_epoch})")

    roc_path = Path(__file__).resolve().parent / "roc_curve.png"
    patient_aucs, slice_aucs = write_roc(
        val,
        roc_path,
        "ED-XAGN one-vs-rest ROC (validation, patient mean)",
    )
    print("  Validation patient ROC (mean pool)")
    print_aucs(patient_aucs, CLASS_NAMES, prefix="  ")
    print("  Validation slice ROC")
    print_aucs(slice_aucs, CLASS_NAMES, prefix="  ")
    print(f"  ROC curve saved to {roc_path}")
    print(f"Classes: {CLASS_NAMES}")

    ckpt_path = Path(__file__).resolve().parent / "edxagn.pt"
    torch.save(best_state, ckpt_path)
    print(f"Saved best weights (epoch {best_epoch}) to {ckpt_path}")


if __name__ == "__main__":
    main()
