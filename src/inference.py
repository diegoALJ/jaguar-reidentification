from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm


@torch.no_grad()
def extract_embeddings(model, loader, device, use_amp=True):
    model.eval()
    embeddings = []
    filenames = []

    for batch in tqdm(loader, desc="Embedding"):
        images = batch["image"].to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            emb, _ = model(images, labels=None)

        embeddings.append(emb.cpu().numpy())
        filenames.extend(batch["filename"])

    embeddings = np.concatenate(embeddings)
    embeddings /= np.clip(
        np.linalg.norm(embeddings, axis=1, keepdims=True),
        1e-12,
        None,
    )

    return pd.DataFrame({"filename": filenames}), embeddings


def ensemble_embeddings(fold_embeddings: list[np.ndarray]) -> np.ndarray:
    ensemble = np.mean(np.stack(fold_embeddings, axis=0), axis=0)
    return ensemble / np.clip(
        np.linalg.norm(ensemble, axis=1, keepdims=True),
        1e-12,
        None,
    )


def build_submission(test_pairs: pd.DataFrame, image_df: pd.DataFrame, embeddings: np.ndarray):
    image_to_idx = {name: i for i, name in enumerate(image_df["filename"])}
    sim_matrix = embeddings @ embeddings.T

    scores = []
    for row in test_pairs.itertuples(index=False):
        q = image_to_idx[row.query_image]
        g = image_to_idx[row.gallery_image]

        score = (float(sim_matrix[q, g]) + 1.0) / 2.0
        scores.append(np.clip(score, 0.0, 1.0))

    return pd.DataFrame({
        "row_id": test_pairs["row_id"],
        "similarity": scores,
    })
