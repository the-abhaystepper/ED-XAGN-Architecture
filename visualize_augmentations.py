"""Show a random MRI with each augmentation from data/dataset.py."""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from PIL import Image
from torchvision import transforms

from data.dataset import CLASS_NAMES, DATASET_NAME, IMG_SIZE


# Geometric augs from BrainTumorDataset(augment=True). Resize/ToTensor are
# preprocessing, so they are applied for display consistency rather than shown
# as separate panels. Flip probability is 1.0 so each effect is visible.
AUGMENTATIONS = {
    "RandomRotation(90)": transforms.RandomRotation(90),
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip(p=1.0),
    "RandomVerticalFlip": transforms.RandomVerticalFlip(p=1.0),
}

COMBINED = transforms.Compose([
    transforms.RandomRotation(90),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])


def pil_to_display(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for image and random augs")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "augmentation_preview.png",
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

    resized = transforms.Resize((IMG_SIZE, IMG_SIZE))(original)

    panels = [("Original", original), (f"Resize({IMG_SIZE})", resized)]
    for name, transform in AUGMENTATIONS.items():
        panels.append((name, transform(resized)))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.0))
    fig.suptitle(f"{DATASET_NAME}\nindex={idx}  class={label_name}", fontsize=12, y=0.98)

    for ax, (title, img) in zip(axes, panels):
        ax.imshow(pil_to_display(img), cmap=None)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{img.size[0]}×{img.size[1]}", fontsize=8, color="0.4")
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
