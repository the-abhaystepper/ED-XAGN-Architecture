import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Figshare Cheng (DOI 10.6084/m9.figshare.1512427): 3064 T1-contrast slices,
# 233 patients, one tumor type per patient. label_id is 1-based.
CLASS_NAMES = ["meningioma", "glioma", "pituitary"]
DATASET_NAME = "MedOtter/Figshare_Brain_Tumor"
IMG_SIZE = 224
BATCH_SIZE = 32

# Published patient-disjoint folds. Fold 1 stays the untouched test set.
# Fold 5 is the early-stopping validation set. Folds 2-4 train.
TRAIN_FOLDS = (2, 3, 4)
VAL_FOLDS = (5,)
TEST_FOLDS = (1,)


def load_figshare():
    """Load every Figshare slice.

    The Hub release stores folds 2-5 as ``train`` and fold 1 as ``test``.
    Concatenating them keeps the published patient fold id on every row.
    """
    bundle = load_dataset(DATASET_NAME)
    if "train" not in bundle or "test" not in bundle:
        raise ValueError(f"{DATASET_NAME} is missing the train or test split")
    return concatenate_datasets([bundle["train"], bundle["test"]])


def slice_to_rgb(image) -> Image.Image:
    """Scale one grayscale MRI slice into an 8-bit RGB image.

    Figshare slices are 16-bit. Per-slice max scaling puts the tensor in [0, 1]
    after ToTensor, which is the range gamma correction expects.
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    peak = float(arr.max())
    if peak > 0:
        arr = arr / peak
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


class BrainTumorDataset(Dataset):
    """Patient-grouped Figshare slices. Training slices are geometrically augmented."""

    def __init__(self, indices: list, augment: bool = False, ds=None):
        self.augment = augment
        self.indices = indices
        self.ds = ds if ds is not None else load_figshare()

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
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
        sample = self.ds[int(self.indices[idx])]
        image = self.transform(slice_to_rgb(sample["image"]))
        label = int(sample["label_id"]) - 1
        pid = str(sample["pid"])
        return image, label, pid


def _column(ds, name: str) -> np.ndarray:
    values = ds[name]
    if name == "pid":
        return np.asarray([str(v) for v in values])
    return np.asarray(values)


def _indices_for(folds: np.ndarray, chosen: tuple) -> list:
    return np.flatnonzero(np.isin(folds, list(chosen))).tolist()


def _patient_labels(pids: np.ndarray, labels: np.ndarray, indices: list) -> dict:
    grouped = {}
    for i in indices:
        pid = pids[i]
        label = int(labels[i])
        previous = grouped.get(pid)
        if previous is None:
            grouped[pid] = label
        elif previous != label:
            raise ValueError(f"Patient {pid} has more than one label ({previous}, {label})")
    return grouped


def class_focal_alpha(labels: np.ndarray, indices: list, alpha: float = 0.25) -> torch.Tensor:
    """Per-class focal alpha from the inverse training-slice frequency.

    The frequency-weighted mean stays at ``alpha``, matching the old scalar.
    Rarer classes get a larger share of the loss.
    """
    counts = np.bincount(labels[np.asarray(indices)], minlength=len(CLASS_NAMES)).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"A training class has no slices: {counts.tolist()}")
    weights = counts.sum() / (len(CLASS_NAMES) * counts) * alpha
    return torch.tensor(weights, dtype=torch.float32)


def _print_split(name: str, patient_labels: dict, indices: list, slice_labels: np.ndarray) -> None:
    print(f"  {name}: {len(patient_labels)} patients, {len(indices)} slices")
    for class_index, class_name in enumerate(CLASS_NAMES):
        n_patients = sum(1 for label in patient_labels.values() if label == class_index)
        n_slices = int(np.sum(slice_labels[indices] == class_index))
        print(f"    {class_name:12s}  patients={n_patients:3d}  slices={n_slices:4d}")


def get_dataloaders() -> tuple:
    ds = load_figshare()
    label_ids = _column(ds, "label_id").astype(np.int64)
    found_ids = set(int(v) for v in label_ids.tolist())
    if found_ids != {1, 2, 3}:
        raise ValueError(f"Expected label_id in {{1, 2, 3}}, found {sorted(found_ids)}")

    names = _column(ds, "label_name")
    for name, label_id in zip(names.tolist(), label_ids.tolist()):
        expected = CLASS_NAMES[int(label_id) - 1]
        if str(name) != expected:
            raise ValueError(f"label_name {name!r} does not match label_id {label_id} ({expected})")

    folds = _column(ds, "fold").astype(np.int64)
    pids = _column(ds, "pid")
    labels = label_ids - 1
    known = set(TRAIN_FOLDS + VAL_FOLDS + TEST_FOLDS)
    found_folds = set(int(v) for v in folds.tolist())
    if found_folds != known:
        raise ValueError(f"Expected folds {sorted(known)}, found {sorted(found_folds)}")

    train_idx = _indices_for(folds, TRAIN_FOLDS)
    val_idx = _indices_for(folds, VAL_FOLDS)
    test_idx = _indices_for(folds, TEST_FOLDS)
    if len(train_idx) + len(val_idx) + len(test_idx) != len(ds):
        raise ValueError("Train, validation, and test do not cover every slice exactly once")

    train_patients = _patient_labels(pids, labels, train_idx)
    val_patients = _patient_labels(pids, labels, val_idx)
    test_patients = _patient_labels(pids, labels, test_idx)
    if not train_patients.keys().isdisjoint(val_patients):
        raise ValueError("A patient is in both train and validation")
    if not train_patients.keys().isdisjoint(test_patients):
        raise ValueError("A patient is in both train and test")
    if not val_patients.keys().isdisjoint(test_patients):
        raise ValueError("A patient is in both validation and test")

    print(
        f"Figshare patient split  train folds {list(TRAIN_FOLDS)}  "
        f"val folds {list(VAL_FOLDS)}  test folds {list(TEST_FOLDS)}"
    )
    _print_split("train", train_patients, train_idx, labels)
    _print_split("val", val_patients, val_idx, labels)
    _print_split("test", test_patients, test_idx, labels)
    class_alpha = class_focal_alpha(labels, train_idx)
    print("  focal alpha by training-slice frequency:")
    for class_name, weight in zip(CLASS_NAMES, class_alpha.tolist()):
        print(f"    {class_name:12s}  alpha={weight:.4f}")

    pin_mem = torch.cuda.is_available()
    train_ds = BrainTumorDataset(train_idx, augment=True, ds=ds)
    val_ds = BrainTumorDataset(val_idx, augment=False, ds=ds)
    test_ds = BrainTumorDataset(test_idx, augment=False, ds=ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_mem)
    return train_loader, val_loader, test_loader, class_alpha
