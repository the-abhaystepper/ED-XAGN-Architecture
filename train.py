import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.models.ed_xagn import EDXAGN
from data.dataset import get_dataloaders, CLASS_NAMES


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:

    ce = F.cross_entropy(logits, targets, reduction='none')
    probs = F.softmax(logits, dim=1)
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    focal = alpha * (1 - pt) ** gamma * ce
    return focal.mean()


def train_one_epoch(model, loader, optimizer, device, epoch: int, total_epochs: int, lambda_aux: float = 0.3):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", unit="batch", dynamic_ncols=True)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        out = model(images, compute_uncertainty=False, return_heatmap=False)
        cls_logits = out['cls_logits']
        aux_logits = out['aux_logits']

        loss_cls = focal_loss(cls_logits, labels)
        loss_aux = focal_loss(aux_logits, labels)
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
            'loss': f"{total_loss / (batch_idx + 1):.4f}",
            'acc': f"{correct / total:.4f}"
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
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_precision': float(precision.mean()),
        'macro_recall': float(recall.mean()),
        'macro_f1': float(f1.mean()),
        'confusion': cm,
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


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, class_names: list, out_path: Path) -> dict:
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
    ax.set_title("ED-XAGN one-vs-rest ROC (validation)")
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
        print(f"  {name:12s}  P={metrics['precision'][i]:.4f}  "
              f"R={metrics['recall'][i]:.4f}  F1={metrics['f1'][i]:.4f}")
    print(f"  {'macro':12s}  P={metrics['macro_precision']:.4f}  "
          f"R={metrics['macro_recall']:.4f}  F1={metrics['macro_f1']:.4f}")


@torch.no_grad()
def validate(model, loader, device, epoch: int, total_epochs: int, lambda_aux: float = 0.3):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Val]  ", unit="batch", dynamic_ncols=True, leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        out = model(images, compute_uncertainty=False, return_heatmap=False)
        cls_logits = out['cls_logits']
        aux_logits = out['aux_logits']

        loss_cls = focal_loss(cls_logits, labels)
        loss_aux = focal_loss(aux_logits, labels)
        loss = loss_cls + lambda_aux * loss_aux

        probs = out['cls_probs']
        preds = cls_logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())

        pbar.set_postfix({
            'val_loss': f"{total_loss / (batch_idx + 1):.4f}",
            'val_acc': f"{correct / total:.4f}"
        })

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()
    metrics = classification_metrics(y_true, y_pred, num_classes=len(CLASS_NAMES))
    return total_loss / len(loader), correct / total, metrics, y_true, y_prob


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders()
    print(f"Training samples: {len(train_loader.dataset)} ({len(train_loader)} batches)")
    print(f"Validation samples: {len(val_loader.dataset)} ({len(val_loader)} batches)")

    print("Building ED-XAGN (loads EfficientNet-B0 + DenseNet-121 ImageNet weights)...", flush=True)
    model = EDXAGN(num_classes=len(CLASS_NAMES))
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"Moving model to {device}...", flush=True)
    model = model.to(device)
    print("Model ready.", flush=True)

    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-6)

    EPOCHS = 2
    lambda_aux = 0.3

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, epoch, EPOCHS, lambda_aux)
        val_loss, val_acc, val_metrics, y_true, y_prob = validate(
            model, val_loader, device, epoch, EPOCHS, lambda_aux
        )
        scheduler.step()
        print(f"Epoch {epoch}/{EPOCHS} complete | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Val P: {val_metrics['macro_precision']:.4f} | "
              f"Val R: {val_metrics['macro_recall']:.4f} | "
              f"Val F1: {val_metrics['macro_f1']:.4f} | "
              f"Time: {time.time()-t0:.1f}s")
        print_metrics(val_metrics, CLASS_NAMES, prefix="  ")

    roc_path = Path(__file__).resolve().parent / "roc_curve.png"
    aucs = plot_roc(y_true, y_prob, CLASS_NAMES, roc_path)

    print(f"\nFinal validation")
    print(f"  Accuracy: {val_acc:.4f}")
    print_metrics(val_metrics, CLASS_NAMES, prefix="  ")
    print("  ROC AUC (one-vs-rest)")
    for name in CLASS_NAMES:
        print(f"    {name:12s}  AUC={aucs[name]:.4f}")
    print(f"    {'macro':12s}  AUC={aucs['macro']:.4f}")
    print(f"  ROC curve saved to {roc_path}")
    print(f"Classes: {CLASS_NAMES}")


if __name__ == "__main__":
    main()
