from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def add_labels_and_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42):
    df = df.copy()

    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df["ground_truth"])
    df["fold"] = -1

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(splitter.split(df, df["label"])):
        df.loc[val_idx, "fold"] = fold

    return df, encoder


def build_transforms(train: bool = True):
    if train:
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03)
            ], p=0.5),
            T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    return T.Compose([
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class JaguarDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row.get("preprocessed_path", row["full_path"])
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "filename": row["filename"],
        }


class JaguarTestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row.get("preprocessed_path", row["full_path"])
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {"image": image, "filename": row["filename"]}


def build_weighted_sampler(df: pd.DataFrame):
    counts = df["label"].value_counts()
    sample_weights = df["label"].map(1.0 / counts).to_numpy()
    return WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_train_valid_loaders(
    df: pd.DataFrame,
    fold: int,
    batch_size: int = 16,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
):
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)

    train_ds = JaguarDataset(train_df, build_transforms(train=True))
    valid_ds = JaguarDataset(valid_df, build_transforms(train=False))

    sampler = build_weighted_sampler(train_df) if use_weighted_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_df, valid_df, train_loader, valid_loader
