"""Shared, leakage-free evaluation utilities for simulated GRN benchmarks.

Matrix convention is regulator x target.  Every method is evaluated on the
same TF-to-gene candidate universe, with self edges excluded.  AUROC and
average precision use continuous scores; F1_max is the maximum F1 on the
precision-recall curve and is reported explicitly as an evaluation summary,
not used for model fitting or hyperparameter selection.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

CELL_TYPE_COUNTS = (2, 3, 4)
SEEDS = (12, 22, 32, 42, 52)
CASE_RE = re.compile(r"^geneNum_(\d+)_id_(\d+)$")

@dataclass(frozen=True, order=True)
class Case:
    gene_num: int
    dataset_id: int
    cell_type_count: int
    seed: int
    data_dir: Path

    @property
    def key(self) -> str:
        return (
            f"geneNum_{self.gene_num}_id_{self.dataset_id}/"
            f"cellType{self.cell_type_count}/seed{self.seed}"
        )

def canonical_labels(values: Iterable[object]) -> list[str]:
    labels = [str(value).strip() for value in values]
    if len(labels) != len(set(labels)):
        raise ValueError("labels must be unique")
    return labels

def enumerate_cases(sim_root: Path) -> list[Case]:
    cases: list[Case] = []
    for dataset_dir in sorted(sim_root.glob("geneNum_*_id_*")):
        match = CASE_RE.match(dataset_dir.name)
        if not match:
            continue
        gene_num, dataset_id = map(int, match.groups())
        for cell_type_count in CELL_TYPE_COUNTS:
            for seed in SEEDS:
                data_dir = dataset_dir / f"cellType{cell_type_count}" / f"seed{seed}"
                if not data_dir.is_dir():
                    raise FileNotFoundError(f"missing benchmark case: {data_dir}")
                cases.append(Case(gene_num, dataset_id, cell_type_count, seed, data_dir))
    if not cases:
        raise FileNotFoundError(f"no simulation cases found below {sim_root}")
    return cases

def read_tf_ids(path: Path) -> list[str]:
    frame = pd.read_csv(path, sep=None, engine="python")
    if frame.shape[1] < 1:
        raise ValueError(f"empty TF file: {path}")
    return canonical_labels(frame.iloc[:, 0])

def read_expression(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", index_col=0)
    frame.index = canonical_labels(frame.index)
    frame.columns = canonical_labels(frame.columns)
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"expression must be finite and non-negative: {path}")
    return frame

def read_metadata(path: Path, expression_cells: Iterable[str]) -> pd.DataFrame:
    metadata = pd.read_csv(path, sep="\t", header=None, names=["cell", "cell_type"], dtype=str)
    metadata["cell"] = metadata["cell"].str.strip()
    metadata["cell_type"] = metadata["cell_type"].str.strip()
    if metadata["cell"].duplicated().any():
        raise ValueError(f"duplicate cells in {path}")
    metadata = metadata.set_index("cell")
    cells = canonical_labels(expression_cells)
    if set(metadata.index) != set(cells):
        missing_meta = sorted(set(cells) - set(metadata.index))[:5]
        missing_expr = sorted(set(metadata.index) - set(cells))[:5]
        raise ValueError(
            f"expression/metadata cell mismatch: missing_meta={missing_meta}, "
            f"missing_expression={missing_expr}"
        )
    return metadata.loc[cells]

def read_truth(path: Path, genes: Iterable[str]) -> pd.DataFrame:
    truth = pd.read_csv(path, sep="\t", index_col=0)
    truth.index = canonical_labels(truth.index)
    truth.columns = canonical_labels(truth.columns)
    genes = canonical_labels(genes)
    if set(truth.index) != set(genes) or set(truth.columns) != set(genes):
        raise ValueError(f"truth/expression gene sets differ: {path}")
    truth = truth.loc[genes, genes]
    values = truth.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"truth contains non-finite values: {path}")
    return (truth != 0).astype(np.uint8)

def candidate_arrays(
    score: pd.DataFrame,
    truth: pd.DataFrame,
    tf_ids: Iterable[str],
) -> tuple[np.ndarray, np.ndarray]:
    score = score.copy()
    score.index = canonical_labels(score.index)
    score.columns = canonical_labels(score.columns)
    genes = list(truth.index)
    if set(score.index) != set(genes) or set(score.columns) != set(genes):
        raise ValueError("score matrix must contain the complete expression gene set")
    score = score.loc[genes, genes]
    tf_ids = canonical_labels(tf_ids)
    unknown_tfs = sorted(set(tf_ids) - set(genes))
    if unknown_tfs:
        raise ValueError(f"TFs absent from expression/truth: {unknown_tfs[:5]}")

    tf_pos = np.array([genes.index(tf) for tf in tf_ids], dtype=int)
    score_values = score.to_numpy(dtype=float)[tf_pos, :]
    truth_values = truth.to_numpy(dtype=np.uint8)[tf_pos, :]
    mask = np.ones(score_values.shape, dtype=bool)
    for row, pos in enumerate(tf_pos):
        mask[row, pos] = False
    y_score = score_values[mask]
    y_true = truth_values[mask]
    if not np.isfinite(y_score).all():
        raise ValueError("score matrix contains non-finite candidate scores")
    if np.unique(y_true).size != 2:
        raise ValueError("candidate truth must contain positive and negative edges")
    return y_true.astype(np.uint8), y_score.astype(float)

def metrics_from_arrays(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | int]:
    auroc = float(roc_auc_score(y_true, y_score))
    average_precision = float(average_precision_score(y_true, y_score))
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    best_index = int(np.argmax(f1))
    threshold = float(thresholds[best_index]) if best_index < len(thresholds) else float("inf")
    return {
        "auroc": auroc,
        "average_precision": average_precision,
        "f1_max": float(f1[best_index]),
        "f1_max_threshold": threshold,
        "n_candidates": int(y_true.size),
        "n_positive": int(y_true.sum()),
    }

def evaluate_matrix(score: pd.DataFrame, truth: pd.DataFrame, tf_ids: Iterable[str]) -> dict[str, float | int]:
    return metrics_from_arrays(*candidate_arrays(score, truth, tf_ids))

def write_json_atomic(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
