"""Shared, label-safe evaluation utilities for simulated CCC benchmarks.

The primary cell-pair evaluation universe is the directed, off-diagonal,
tie-inclusive KNN graph used by TMENet (K=4 for the canonical simulations).
Restricting every cell-resolved method to this same universe avoids inflating
AUROC with spatially impossible pairs and prevents row-order mismatches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

def canonical_labels(values: Iterable[object]) -> list[str]:
    labels = [str(value).strip() for value in values]
    if len(labels) != len(set(labels)):
        raise ValueError("labels must be unique")
    return labels

def read_labeled_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", index_col=0)
    frame.index = canonical_labels(frame.index)
    frame.columns = canonical_labels(frame.columns)
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"matrix contains non-finite values: {path}")
    return frame

def read_case_cell_order(case_dir: Path) -> list[str]:
    # A stacks one gene block per cell type, so repeated row labels are
    # expected.  Only its columns define the canonical cell order.
    header = pd.read_csv(case_dir / "A.txt", sep="\t", index_col=0, nrows=0)
    return canonical_labels(header.columns)

def align_square(frame: pd.DataFrame, labels: Iterable[str], name: str) -> pd.DataFrame:
    labels = canonical_labels(labels)
    if frame.shape[0] != frame.shape[1]:
        raise ValueError(f"{name} must be square, got {frame.shape}")
    if set(frame.index) != set(labels) or set(frame.columns) != set(labels):
        raise ValueError(f"{name} label set differs from the canonical cell set")
    return frame.loc[labels, labels]

def tie_inclusive_knn(distance: pd.DataFrame, labels: Iterable[str], k: int) -> np.ndarray:
    labels = canonical_labels(labels)
    aligned = align_square(distance, labels, "distance").to_numpy(dtype=float)
    if (aligned < 0).any():
        raise ValueError("distance matrix contains negative values")
    safe = aligned.copy()
    np.fill_diagonal(safe, np.inf)
    k_effective = min(int(k), len(labels) - 1)
    if k_effective < 1:
        raise ValueError("at least two cells are required")
    kth_distance = np.sort(safe, axis=1)[:, k_effective - 1]
    mask = safe <= kth_distance[:, None]
    np.fill_diagonal(mask, False)
    return mask

def load_case_truth_and_candidates(
    case_dir: Path,
    k: int = 4,
) -> tuple[list[str], pd.DataFrame, np.ndarray]:
    labels = read_case_cell_order(case_dir)
    distance = read_labeled_matrix(case_dir / "D.txt")
    truth = align_square(read_labeled_matrix(case_dir / "gold_CCC.txt"), labels, "gold_CCC")
    truth = (truth != 0).astype(np.uint8)
    candidate_mask = tie_inclusive_knn(distance, labels, k)
    outside = int(np.count_nonzero(truth.to_numpy(dtype=bool) & ~candidate_mask))
    if outside:
        raise ValueError(f"gold_CCC has {outside} edges outside the KNN candidate universe")
    return labels, truth, candidate_mask

def candidate_arrays(
    score: pd.DataFrame,
    truth: pd.DataFrame,
    candidate_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = list(truth.index)
    aligned_score = align_square(score, labels, "score")
    if candidate_mask.shape != truth.shape:
        raise ValueError("candidate mask shape differs from the truth matrix")
    mask = np.asarray(candidate_mask, dtype=bool).copy()
    np.fill_diagonal(mask, False)
    y_true = truth.to_numpy(dtype=np.uint8)[mask]
    y_score = aligned_score.to_numpy(dtype=float)[mask]
    if not np.isfinite(y_score).all():
        raise ValueError("score matrix contains non-finite candidate scores")
    if np.unique(y_true).size != 2:
        raise ValueError("candidate truth must contain both positive and negative edges")
    return y_true, y_score

def metrics_from_arrays(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | int]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.argmax(f1))
    threshold = float(thresholds[best_index]) if best_index < len(thresholds) else float("inf")
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "f1_max": float(f1[best_index]),
        "f1_max_threshold": threshold,
        "n_candidates": int(y_true.size),
        "n_positive": int(y_true.sum()),
        "positive_fraction": float(np.mean(y_true)),
    }

def evaluate_cell_pair_matrix(
    score: pd.DataFrame,
    truth: pd.DataFrame,
    candidate_mask: np.ndarray,
) -> dict[str, float | int]:
    return metrics_from_arrays(*candidate_arrays(score, truth, candidate_mask))
