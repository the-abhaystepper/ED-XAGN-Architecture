import argparse
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torchvision import transforms

from data.dataset import CLASS_NAMES, DATASET_NAME, IMG_SIZE
from src.models.ed_xagn import EDXAGN

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "edxagn.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for image selection")
    parser.add_argument("--index", type=int, default=None, help="Dataset index (overrides random draw)")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to EDXAGN state_dict")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "classification_preview.png",
        help="Where to save the figure",
    )
    parser.add_argument("--no-show", action="store_true", help="Save the figure without opening a window")
    return parser.parse_args()


def load_model(weights_path: Path, device: torch.device) -> EDXAGN:
    model = EDXAGN(num_classes=len(CLASS_NAMES))
    if weights_path.is_file():
        try:
            state = torch.load(weights_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"No checkpoint at {weights_path}; using ImageNet backbones and randomly initialized heads")
    model.to(device)
    model.eval()
    return model


def overlay_heatmap(image: torch.Tensor, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib.pyplot as plt

    img = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    heat = plt.get_cmap("jet")(heatmap)[..., :3]
    return np.clip((1.0 - alpha) * img + alpha * heat, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    ds = load_dataset(DATASET_NAME, split="train")
    idx = args.index if args.index is not None else random.randrange(len(ds))
    sample = ds[idx]
    original = sample["image"].convert("RGB")
    true_idx = int(sample["label"])
    true_name = CLASS_NAMES[true_idx]

    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    image = preprocess(original)

    model = load_model(args.weights, device)
    with torch.set_grad_enabled(True):
        out = model(image.unsqueeze(0).to(device), compute_uncertainty=True, return_heatmap=True)

    probs = out["cls_probs"][0].detach().cpu()
    pred_idx = int(probs.argmax().item())
    pred_name = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx].item())
    uncertainty = float(out["uncertainty"][0].detach().cpu().item())
    heatmap = out["heatmap"][0, 0].detach().cpu().numpy()
    correct = pred_idx == true_idx

    print(f"\nSample index : {idx}")
    print(f"Actual label : {true_name} ({true_idx})")
    print(f"Predicted    : {pred_name} ({pred_idx})  p={confidence:.4f}")
    print(f"Uncertainty  : {uncertainty:.6f}  (MC dropout variance)")
    print(f"Match        : {'YES' if correct else 'NO'}")
    print("Class probabilities:")
    for i, name in enumerate(CLASS_NAMES):
        marker = " <=" if i == pred_idx else ""
        print(f"  {name:12s}  {probs[i].item():.4f}{marker}")

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_np = image.permute(1, 2, 0).numpy()
    overlay = overlay_heatmap(image, heatmap)
    verdict = "correct" if correct else "incorrect"
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    fig.suptitle(
        f"index={idx}  pred={pred_name} ({confidence:.2f})  "
        f"true={true_name}  [{verdict}]  uncertainty={uncertainty:.4f}",
        fontsize=11,
        y=0.98,
    )

    panels = [
        ("Original", img_np, None),
        ("Grad-CAM", heatmap, "jet"),
        ("Overlay", overlay, None),
    ]
    for ax, (title, data, cmap) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if cmap == "jet":
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved {args.out}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
