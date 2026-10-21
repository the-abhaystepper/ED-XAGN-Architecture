"""Show a random MRI before and after gamma normalization."""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torchvision import transforms

from data.dataset import CLASS_NAMES, DATASET_NAME, IMG_SIZE
from src.utils.gamma import gamma_correction

GAMMA = 1.2
A = 1.0


def tensor_to_display(img: torch.Tensor) -> np.ndarray:
    return img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def pil_to_display(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for image selection")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="Gamma exponent (paper default 1.2)")
    parser.add_argument("--A", type=float, default=A, dest="scale", help="Scaling factor A (paper default 1.0)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "gamma_preview.png",
        help="Where to save the figure",
    )
    parser.add_argument("--no-show", action="store_true", help="Save the figure without opening a window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    ds = load_dataset(DATASET_NAME, split="train")
    idx = random.randrange(len(ds))
    sample = ds[idx]
    original = sample["image"].convert("RGB")
    label_name = CLASS_NAMES[sample["label"]]

    to_tensor = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    raw = to_tensor(original)
    normalized = gamma_correction(raw.unsqueeze(0), gamma=args.gamma, A=args.scale).squeeze(0)

    panels = [
        ("Original", pil_to_display(original), f"{original.size[0]}×{original.size[1]}"),
        (f"Resize({IMG_SIZE})  I_raw", tensor_to_display(raw), f"{IMG_SIZE}×{IMG_SIZE}"),
        (f"Gamma  A={args.scale}  γ={args.gamma}", tensor_to_display(normalized), f"{IMG_SIZE}×{IMG_SIZE}"),
    ]

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.0))
    fig.suptitle(f"{DATASET_NAME}\nindex={idx}  class={label_name}", fontsize=12, y=0.98)

    for ax, (title, img, size_label) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(size_label, fontsize=8, color="0.4")
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.out}")
    print(f"Sample index={idx}  label={sample['label']} ({label_name})")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
