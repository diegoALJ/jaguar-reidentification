from __future__ import annotations

import numpy as np
import pandas as pd


def average_precision(relevant_mask: np.ndarray) -> float:
    relevant_mask = relevant_mask.astype(np.int32)
    n_relevant = relevant_mask.sum()

    if n_relevant == 0:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for rank, rel in enumerate(relevant_mask, start=1):
        if rel:
            hits += 1
            precision_sum += hits / rank

    return float(precision_sum / n_relevant)


def identity_balanced_map(embeddings: np.ndarray, labels: np.ndarray) -> float:
    embeddings = embeddings / np.clip(
        np.linalg.norm(embeddings, axis=1, keepdims=True),
        1e-12,
        None,
    )
    similarities = embeddings @ embeddings.T

    rows = []

    for i in range(len(labels)):
        sims = similarities[i].copy()
        sims[i] = -np.inf
        order = np.argsort(-sims)
        rel = labels[order] == labels[i]

        rows.append({"label": labels[i], "ap": average_precision(rel)})

    return float(pd.DataFrame(rows).groupby("label")["ap"].mean().mean())
