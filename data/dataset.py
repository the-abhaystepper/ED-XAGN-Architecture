import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import transforms
import numpy as np


CLASS_NAMES = ["no_tumor", "glioma", "meningioma", "pituitary"]
DATASET_NAME = "BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset"
IMG_SIZE = 224
BATCH_SIZE = 32


class BrainTumorDataset(Dataset):
    """Brain tumor MRI dataset with rescaling and optional augmentations."""

    def __init__(self, indices: list, augment: bool = False, ds=None):
        self.augment = augment
        self.indices = indices
        self.ds = ds if ds is not None else load_dataset(DATASET_NAME, split="train")

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple:
        orig_idx = self.indices[idx]
        sample = self.ds[orig_idx]
        img = sample["image"].convert("RGB")
        img = self.transform(img)
        label = sample["label"]
        return img, label


def get_dataloaders() -> tuple:
    ds = load_dataset(DATASET_NAME, split="train")
    n = len(ds)

    np.random.seed(42)
    perm = np.random.permutation(n)
    n_val = n // 5  # 20% validation
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_ds = BrainTumorDataset(train_idx.tolist(), augment=True, ds=ds)
    val_ds = BrainTumorDataset(val_idx.tolist(), augment=False, ds=ds)

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_mem)
    return train_loader, val_loader
