import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from configs.config import ACTIVE_CLASSES, EFFICIENTNET_IMG_SIZE, EFFICIENTNET_BATCH

def get_transforms(split: str = "train", img_size: int = EFFICIENTNET_IMG_SIZE):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

class RiceDiseaseDataset(Dataset):
    def __init__(
        self,
        root_dir: str = None,
        csv_path: str = None,
        split: str = "train",
        img_size: int = EFFICIENTNET_IMG_SIZE,
        class_names: List[str] = None,
    ):
        self.transform = get_transforms(split, img_size)
        self.class_names = class_names or ACTIVE_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        if csv_path:
            df = pd.read_csv(csv_path)
            self.samples = list(zip(df["image_path"].tolist(), df["label"].tolist()))
        elif root_dir:
            self.samples = self._scan_directory(Path(root_dir))
        else:
            raise ValueError("Either root_dir or csv_path must be provided")

    def _scan_directory(self, root: Path) -> List[Tuple[str, int]]:
        samples = []
        for cls_name in self.class_names:
            cls_dir = root / cls_name
            if not cls_dir.exists():
                continue
            idx = self.class_to_idx[cls_name]
            for img_path in cls_dir.glob("**/*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    samples.append((str(img_path), idx))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (EFFICIENTNET_IMG_SIZE, EFFICIENTNET_IMG_SIZE))
        return self.transform(img), label

    def get_class_weights(self) -> torch.Tensor:
        counts = np.zeros(len(self.class_names))
        for _, label in self.samples:
            counts[label] += 1
        counts = np.maximum(counts, 1)
        weights = 1.0 / counts
        weights = weights / weights.sum()
        return torch.FloatTensor(weights)

    def get_sampler(self) -> WeightedRandomSampler:
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label] for _, label in self.samples]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

def build_dataloaders(
    train_dir: str,
    val_dir: str,
    test_dir: str = None,
    batch_size: int = EFFICIENTNET_BATCH,
    num_workers: int = 4,
    use_sampler: bool = True,
) -> dict:
    train_ds = RiceDiseaseDataset(root_dir=train_dir, split="train")
    val_ds = RiceDiseaseDataset(root_dir=val_dir, split="val")

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_ds.get_sampler() if use_sampler else None,
            shuffle=False if use_sampler else True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    if test_dir and Path(test_dir).exists():
        test_ds = RiceDiseaseDataset(root_dir=test_dir, split="test")
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        print(f"DataLoaders ready: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    else:
        print(f"DataLoaders ready: train={len(train_ds)}, val={len(val_ds)}")

    return loaders