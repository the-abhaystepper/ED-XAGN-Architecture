import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

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


@torch.no_grad()
def validate(model, loader, device, epoch: int, total_epochs: int, lambda_aux: float = 0.3):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Val]  ", unit="batch", dynamic_ncols=True, leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        out = model(images, compute_uncertainty=False, return_heatmap=False)
        cls_logits = out['cls_logits']
        aux_logits = out['aux_logits']

        loss_cls = focal_loss(cls_logits, labels)
        loss_aux = focal_loss(aux_logits, labels)
        loss = loss_cls + lambda_aux * loss_aux

        preds = cls_logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

        pbar.set_postfix({
            'val_loss': f"{total_loss / (batch_idx + 1):.4f}",
            'val_acc': f"{correct / total:.4f}"
        })

    return total_loss / len(loader), correct / total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders()
    print(f"Training samples: {len(train_loader.dataset)} ({len(train_loader)} batches)")
    print(f"Validation samples: {len(val_loader.dataset)} ({len(val_loader)} batches)")

    model = EDXAGN(num_classes=len(CLASS_NAMES)).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-6)

    EPOCHS = 1
    lambda_aux = 0.3

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, epoch, EPOCHS, lambda_aux)
        val_loss, val_acc = validate(model, val_loader, device, epoch, EPOCHS, lambda_aux)
        scheduler.step()
        print(f"Epoch {epoch}/{EPOCHS} complete | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {time.time()-t0:.1f}s")

    print(f"\nFinal Validation Accuracy: {val_acc:.4f}")
    print(f"Classes: {CLASS_NAMES}")


if __name__ == "__main__":
    main()
