"""Unchanged matrix-loading and alignment helpers extracted from reference scripts."""
import numpy as np
import pandas as pd


def load_matrix(path):

    df = pd.read_csv(path, sep='\t', index_col=0)

    return df.values, list(df.index), list(df.columns)


def build_blockdiag_B_mask(gene_num: int, cell_type_count: int) -> np.ndarray:
    m = gene_num * cell_type_count

    B_mask = np.zeros((m, m), dtype=float)

    for i in range(cell_type_count):

        s = i * gene_num

        e = s + gene_num

        B_mask[s:e, s:e] = 1.0

    return B_mask


def align_distance_and_knn(D, row_labels, col_labels, target_labels, k):
    """Build a tie-inclusive KNN mask, then align it with A/C.

    All cells at the Kth-neighbour distance are retained.  This makes the mask
    invariant to label reordering and avoids arbitrarily rejecting equidistant
    spatial neighbours.
    """
    rows = [str(label) for label in row_labels]
    cols = [str(label) for label in col_labels]
    target = [str(label) for label in target_labels]
    if D.shape[0] != D.shape[1] or len(rows) != D.shape[0] or len(cols) != D.shape[1]:
        raise ValueError("D must be square and have complete row/column labels")
    if len(set(rows)) != len(rows) or len(set(cols)) != len(cols) or len(set(target)) != len(target):
        raise ValueError("Cell labels must be unique")
    if set(rows) != set(cols) or set(rows) != set(target):
        raise ValueError("A/C and D cell-label sets do not match")

    safe = np.asarray(D, dtype=float).copy()
    row_to_col = np.array([cols.index(label) for label in rows])
    safe[np.arange(len(rows)), row_to_col] = np.inf
    k_eff = min(int(k), len(rows) - 1)
    kth_distance = np.sort(safe, axis=1)[:, k_eff - 1]
    source_mask = (safe <= kth_distance[:, None]).astype(float)
    source_mask[np.arange(len(rows)), row_to_col] = 0.0

    row_pos = {label: i for i, label in enumerate(rows)}
    col_pos = {label: i for i, label in enumerate(cols)}
    row_idx = [row_pos[label] for label in target]
    col_idx = [col_pos[label] for label in target]
    return D[np.ix_(row_idx, col_idx)], source_mask[np.ix_(row_idx, col_idx)]
