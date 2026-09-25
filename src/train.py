from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .metrics import identity_balanced_map


def train_one_epoch(model, loader, optimizer, scheduler, device, use_amp=True):
    model.train()
    scaler = GradScaler(enabled=use_amp)
    running_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            _, logits = model(images, labels)
            loss = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, device, use_amp=True):
    model.eval()

    embeddings = []
    labels = []

    for batch in tqdm(loader, desc="Valid", leave=False):
        images = batch["image"].to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            emb, _ = model(images, labels=None)

        embeddings.append(emb.cpu().numpy())
        labels.append(batch["label"].numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)

    return identity_balanced_map(embeddings, labels), embeddings, labels


def save_checkpoint(model, path: str | Path, fold: int, score: float):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "fold": fold,
            "best_score": float(score),
        },
        path,
    )
