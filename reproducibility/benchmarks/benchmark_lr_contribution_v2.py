#!/usr/bin/env python3
"""Benchmark post-hoc LR contributions on scMultiSim simulations.

The common native unit is sender cell type x receiver cell type x LR pair.
The LR direction follows the simulator exactly: regulator=ligand and
target=receptor.  Truth is constructed from the realized per-cell CCI file,
while the supplied continuous effect is retained as an independent strength
target.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (
    average_precision_score,
    ndcg_score,
    precision_recall_curve,
    roc_auc_score,
)

from ccc_benchmark_common import align_square, read_labeled_matrix, tie_inclusive_knn

CASE_PATTERN = re.compile(r"^geneNum_([0-9]+)_id_([0-9]+)/cellType([0-9]+)/seed([0-9]+)$")

def gene_label(value: object) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text[1:] if text.startswith("g") else text

def read_metadata(case_dir: Path, truth_dir: Path | None = None) -> pd.DataFrame:
    metadata_path = case_dir / "metadata_fine.tsv"
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path, sep="\t", dtype=str)
        if not {"cell", "cell_type"}.issubset(metadata.columns):
            raise ValueError("case metadata_fine.tsv lacks cell/cell_type")
        metadata = metadata[["cell", "cell_type"]]
    elif truth_dir is None:
        metadata = pd.read_csv(
            case_dir / "metadata.txt", sep="\t", header=None,
            names=["cell", "cell_type"], dtype=str,
        )
    else:
        metadata_path = truth_dir / "metadata_fine.tsv"
        if not metadata_path.exists():
            metadata_path = truth_dir / "cell_metadata.tsv"
        metadata = pd.read_csv(metadata_path, sep="\t", dtype=str)
        if not {"cell", "cell_type"}.issubset(metadata.columns):
            raise ValueError("fine truth metadata lacks cell/cell_type")
        metadata = metadata[["cell", "cell_type"]]
    metadata["cell"] = metadata["cell"].str.strip()
    metadata["cell_type"] = metadata["cell_type"].str.strip()
    if metadata["cell"].duplicated().any():
        raise ValueError("metadata contains duplicate cells")
    return metadata.set_index("cell")

def read_expression(case_dir: Path, cells: list[str]) -> pd.DataFrame:
    counts = pd.read_csv(case_dir / "rnaseq_data.txt", sep="\t", index_col=0)
    counts.index = [gene_label(value) for value in counts.index]
    counts.columns = [str(value).strip() for value in counts.columns]
    if counts.index.duplicated().any():
        counts = counts.groupby(level=0, sort=False).sum()
    if set(counts.columns) != set(cells):
        raise ValueError("expression and metadata cell labels differ")
    counts = counts.loc[:, cells].astype(float)
    library_size = counts.sum(axis=0).replace(0, 1)
    return np.log1p(counts.divide(library_size, axis=1) * 10000.0)

def read_raw_expression(case_dir: Path, cells: list[str]) -> pd.DataFrame:
    counts = pd.read_csv(case_dir / "rnaseq_data.txt", sep="\t", index_col=0)
    counts.index = [gene_label(value) for value in counts.index]
    counts.columns = [str(value).strip() for value in counts.columns]
    if counts.index.duplicated().any():
        counts = counts.groupby(level=0, sort=False).sum()
    if set(counts.columns) != set(cells):
        raise ValueError("expression and metadata cell labels differ")
    return counts.loc[:, cells].astype(float)

def read_lr_network(
    case_dir: Path, lr_root: Path, gene_num: int, dataset_id: int
) -> pd.DataFrame:
    path = case_dir / "LR_candidate_universe.tsv"
    if path.exists():
        frame = pd.read_csv(path, sep="\t")
    else:
        path = lr_root / f"geneNum_{gene_num}_{dataset_id}.txt"
        frame = pd.read_csv(path, sep="\t")
    names = {str(column).strip().lower(): column for column in frame.columns}
    if {"ligand", "receptor"}.issubset(names):
        ligand_column, receptor_column = names["ligand"], names["receptor"]
    elif {"regulator", "target"}.issubset(names):
        ligand_column, receptor_column = names["regulator"], names["target"]
    else:
        raise ValueError(f"CCI candidates lack ligand/receptor identities: {path}")
    effect = (
        pd.to_numeric(frame[names["effect"]], errors="raise")
        if "effect" in names else pd.Series(0.0, index=frame.index)
    )
    result = pd.DataFrame({
        "ligand": frame[ligand_column].map(gene_label),
        "receptor": frame[receptor_column].map(gene_label),
        "effect": effect,
    })
    # scMultiSim's .parseSpatialParams iterates over input rows and assigns
    # effects[target, regulator] <- weight.  Therefore a duplicated LR is
    # governed by its last row, rather than the mean or sum of duplicate rows.
    duplicate_count = int(result.duplicated(["ligand", "receptor"]).sum())
    result = result.drop_duplicates(["ligand", "receptor"], keep="last").reset_index(drop=True)
    result.attrs["input_row_count"] = int(len(frame))
    result.attrs["duplicate_lr_row_count"] = duplicate_count
    result.attrs["candidate_source"] = str(path)
    result.attrs["contains_false_lr_candidates"] = bool(
        "truth" in names and (pd.to_numeric(frame[names["truth"]]) == 0).any()
    )
    if ((result["effect"] < 0) | (result["effect"] > 1)).any():
        raise ValueError("CCI effects must lie in [0, 1]")
    return result

def load_coupling_score(path: Path, cells: list[str], method: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        score = np.asarray(data["score"], dtype=float)
        labels = [str(value).strip() for value in data["cells"]]
    if score.shape != (len(labels), len(labels)) or set(labels) != set(cells):
        raise ValueError(f"{method} coupling labels or shape differ from metadata")
    order = [labels.index(cell) for cell in cells]
    score = score[np.ix_(order, order)]
    if not np.isfinite(score).all():
        raise ValueError(f"{method} coupling contains non-finite values")
    if score.min() < -1e-8:
        raise ValueError(f"{method} coupling contains negative values (min={score.min():.4g})")
    return np.maximum(score, 0)

def build_truth(
    case_dir: Path,
    metadata: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    candidate_mask: np.ndarray,
    seed: int,
    truth_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    cells = metadata.index.tolist()
    cell_position = {cell: index for index, cell in enumerate(cells)}
    exact_case_path = case_dir / "realized_lr_edges.tsv"
    if exact_case_path.exists():
        realized_path = exact_case_path
        truth_source = "exact_case_metadata_fine_and_realized_cci_gt"
        reverse_legacy_direction = False
    elif truth_dir is not None:
        realized_path = truth_dir / "realized_lr_edges.tsv"
        truth_source = "external_fine_state_export"
        reverse_legacy_direction = False
    else:
        realized_path = case_dir.parent / "CCI" / f"seed{seed}_cci.txt"
        truth_source = "legacy_export_with_direction_correction"
        reverse_legacy_direction = True
    realized = pd.read_csv(realized_path, sep="\t", dtype=str)
    required = {"sender", "receiver", "ligand", "receptor"}
    if not required.issubset(realized.columns):
        raise ValueError(f"realized CCI lacks {sorted(required)}: {realized_path}")
    if reverse_legacy_direction:
        # The legacy exporter reversed ct1/ct2. The grid is undirected, so
        # swapping stored identities reconstructs the simulated direction.
        realized[["sender", "receiver"]] = realized[["receiver", "sender"]].to_numpy()
    realized["sender"] = realized["sender"].str.strip()
    realized["receiver"] = realized["receiver"].str.strip()
    realized["ligand"] = realized["ligand"].map(gene_label)
    realized["receptor"] = realized["receptor"].map(gene_label)
    missing_cells = (set(realized["sender"]) | set(realized["receiver"])) - set(cells)
    if missing_cells:
        raise ValueError(f"realized CCI contains unknown cells: {sorted(missing_cells)[:5]}")
    known_lr = set(zip(lr_pairs["ligand"], lr_pairs["receptor"]))
    unknown_lr = set(zip(realized["ligand"], realized["receptor"])) - known_lr
    if unknown_lr:
        raise ValueError(f"realized CCI contains LR pairs absent from the supplied network: {list(unknown_lr)[:5]}")

    sender_pos = realized["sender"].map(cell_position).to_numpy()
    receiver_pos = realized["receiver"].map(cell_position).to_numpy()
    inside = candidate_mask[sender_pos, receiver_pos]
    if not inside.all():
        raise ValueError(f"{int((~inside).sum())} realized CCI edges fall outside directed KNN candidates")
    realized_row_count = int(len(realized))
    realized = realized.drop_duplicates(["sender", "receiver", "ligand", "receptor"]).copy()
    realized["sender_type"] = realized["sender"].map(metadata["cell_type"])
    realized["receiver_type"] = realized["receiver"].map(metadata["cell_type"])
    realized_counts = (
        realized.groupby(["sender_type", "receiver_type", "ligand", "receptor"])
        .size().rename("truth_edge_count").reset_index()
    )

    type_values = sorted(metadata["cell_type"].unique())
    type_array = metadata["cell_type"].to_numpy()
    type_pairs = []
    for sender_type in type_values:
        for receiver_type in type_values:
            eligible = candidate_mask & (type_array[:, None] == sender_type) & (type_array[None, :] == receiver_type)
            count = int(eligible.sum())
            if count:
                type_pairs.append((sender_type, receiver_type, count))
    universe = pd.DataFrame(type_pairs, columns=["sender_type", "receiver_type", "eligible_edge_count"])
    universe = universe.merge(lr_pairs, how="cross")
    universe = universe.merge(
        realized_counts,
        on=["sender_type", "receiver_type", "ligand", "receptor"],
        how="left",
    )
    universe["truth_edge_count"] = universe["truth_edge_count"].fillna(0).astype(int)
    universe["truth_present"] = (universe["truth_edge_count"] > 0).astype(int)
    universe["truth_density"] = universe["truth_edge_count"] / universe["eligible_edge_count"]
    universe["truth_effect_density"] = universe["effect"] * universe["truth_density"]
    audit = {
        "truth_source": truth_source,
        "realized_edge_rows_before_deduplication": realized_row_count,
        "realized_duplicate_edge_rows": int(realized_row_count - len(realized)),
        "realized_edge_rows": int(len(realized)),
        "realized_unique_cell_lr_edges": int(realized[["sender", "receiver", "ligand", "receptor"]].drop_duplicates().shape[0]),
        "positive_typepair_lr": int(universe["truth_present"].sum()),
        "candidate_typepair_lr": int(len(universe)),
    }
    return universe, audit

def positive_rank_expression(expression: pd.DataFrame) -> pd.DataFrame:
    """Gene-wise percentile ranks with observed zeros kept at zero."""
    result = pd.DataFrame(0.0, index=expression.index, columns=expression.columns)
    for gene in expression.index:
        values = expression.loc[gene]
        positive = values > 0
        if positive.any():
            result.loc[gene, positive] = values.loc[positive].rank(
                method="average", pct=True
            )
    return result

def add_conserved_coupling_attribution(
    table: pd.DataFrame,
    expression: pd.DataFrame,
    coupling: np.ndarray,
    metadata: pd.DataFrame,
    candidate_mask: np.ndarray,
    prefix: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Allocate each learned cell-pair coupling across competing LR pairs.

    For candidate edge i->j, positive-rank LR support q is normalized over all
    supplied LR pairs and multiplied by coupling C_ij.  Consequently, LR
    attributions sum to C_ij whenever at least one LR has expression support.
    """
    result = table.copy()
    cells = metadata.index.tolist()
    type_array = metadata["cell_type"].to_numpy()
    ranked = positive_rank_expression(expression)
    gene_position = {gene: index for index, gene in enumerate(ranked.index)}
    expression_values = ranked.to_numpy(dtype=float)
    missing = (set(result["ligand"]) | set(result["receptor"])) - set(gene_position)
    if missing:
        raise ValueError(f"LR genes absent from expression: {sorted(missing)[:5]}")

    lr_pairs = result[["ligand", "receptor"]].drop_duplicates().reset_index(drop=True)
    lr_index = {
        (row.ligand, row.receptor): index
        for index, row in enumerate(lr_pairs.itertuples(index=False))
    }
    sender_pos, receiver_pos = np.where(candidate_mask)
    if not len(sender_pos):
        raise ValueError("empty candidate edge universe")
    ligand_pos = np.asarray([gene_position[value] for value in lr_pairs["ligand"]])
    receptor_pos = np.asarray([gene_position[value] for value in lr_pairs["receptor"]])
    ligand_values = expression_values[ligand_pos[:, None], sender_pos[None, :]]
    receptor_values = expression_values[receptor_pos[:, None], receiver_pos[None, :]]
    support = ligand_values * receptor_values
    support_sum = support.sum(axis=0)
    shares = np.divide(
        support,
        support_sum[None, :],
        out=np.zeros_like(support),
        where=support_sum[None, :] > 0,
    )
    edge_coupling = coupling[sender_pos, receiver_pos]
    attribution = shares * edge_coupling[None, :]
    weighted_support = support * edge_coupling[None, :]
    sender_types = type_array[sender_pos]
    receiver_types = type_array[receiver_pos]

    primary = []
    raw_support = []
    c_times_support = []
    c_mean = []
    for row in result.itertuples(index=False):
        edge_mask = (sender_types == row.sender_type) & (receiver_types == row.receiver_type)
        if not edge_mask.any():
            raise ValueError("truth universe contains a type pair without candidate edges")
        pair_index = lr_index[(row.ligand, row.receptor)]
        primary.append(float(attribution[pair_index, edge_mask].mean()))
        raw_support.append(float(support[pair_index, edge_mask].mean()))
        c_times_support.append(float(weighted_support[pair_index, edge_mask].mean()))
        c_mean.append(float(edge_coupling[edge_mask].mean()))
    result[f"{prefix}_lr_conserved_attribution"] = primary
    result[f"{prefix}_c_times_lr_support"] = c_times_support
    # Store expression-only support once; it is identical for all couplings.
    if "expression_lr_rank_support" not in result:
        result["expression_lr_rank_support"] = raw_support
    result[f"{prefix}_coupling_mean"] = c_mean

    reconstructed = attribution.sum(axis=0)
    expected = edge_coupling * (support_sum > 0)
    residual = np.abs(reconstructed - expected)
    audit = {
        f"{prefix}_candidate_cell_edges": int(len(sender_pos)),
        f"{prefix}_edges_with_lr_support": int((support_sum > 0).sum()),
        f"{prefix}_max_cell_edge_conservation_error": float(residual.max()),
        f"{prefix}_mean_cell_edge_conservation_error": float(residual.mean()),
    }
    if residual.max() > 1e-8:
        raise ValueError(f"{prefix} LR attribution fails coupling conservation")
    return result, audit

def add_mdic3_lr_official(
    table: pd.DataFrame,
    raw_expression: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Reproduce MDIC3_LR.py's expression-correlation significance score.

    The official post-hoc script does not use MDIC3's CCC matrix.  For each
    sender/receiver type it independently sorts ligand and receptor values,
    truncates to equal length, computes Pearson P values, and applies
    Bonferroni correction only to pairs with raw P <= 0.05.  Pairs surviving
    adjusted P <= 0.01 receive -log10(adjusted P); all others receive zero.
    """
    result = table.copy()
    type_values = list(dict.fromkeys(metadata["cell_type"].tolist()))
    type_cells = {
        value: metadata.index[metadata["cell_type"] == value].tolist()
        for value in type_values
    }
    lr_genes = set(result["ligand"]) | set(result["receptor"])
    eligible_genes = set()
    # Preserve the official script's cumulative, type-ordered ubiquity filter.
    for gene in lr_genes:
        if gene not in raw_expression.index:
            continue
        accumulated: list[float] = []
        ubiquitous_steps = 0
        for cell_type in type_values:
            accumulated.extend(raw_expression.loc[gene, type_cells[cell_type]].tolist())
            if sum(value > 0 for value in accumulated) > len(accumulated) / 2:
                ubiquitous_steps += 1
        if ubiquitous_steps < len(type_values) / 2:
            eligible_genes.add(gene)

    scores = np.zeros(len(result), dtype=float)
    raw_correlations = np.zeros(len(result), dtype=float)
    raw_pvalues = np.ones(len(result), dtype=float)
    group_keys = ["sender_type", "receiver_type"]
    for _, indices in result.groupby(group_keys, sort=False).groups.items():
        tested = []
        for index in indices:
            row = result.loc[index]
            ligand, receptor = row["ligand"], row["receptor"]
            if ligand not in eligible_genes or receptor not in eligible_genes:
                continue
            ligand_values = sorted(
                raw_expression.loc[ligand, type_cells[row["sender_type"]]].tolist(),
                reverse=True,
            )
            receptor_values = sorted(
                raw_expression.loc[receptor, type_cells[row["receiver_type"]]].tolist(),
                reverse=True,
            )
            if sum(value == 0 for value in ligand_values) > 0.7 * len(ligand_values):
                continue
            if sum(value == 0 for value in receptor_values) > 0.7 * len(receptor_values):
                continue
            size = min(len(ligand_values), len(receptor_values))
            x = np.asarray(ligand_values[:size], dtype=float)
            y = np.asarray(receptor_values[:size], dtype=float)
            if size < 2 or np.std(x) == 0 or np.std(y) == 0:
                continue
            correlation, pvalue = pearsonr(x, y)
            if not np.isfinite(correlation) or not np.isfinite(pvalue):
                continue
            raw_correlations[index] = float(correlation)
            raw_pvalues[index] = float(pvalue)
            if pvalue <= 0.05:
                tested.append(index)
        correction_count = len(tested)
        for index in tested:
            adjusted = min(1.0, raw_pvalues[index] * correction_count)
            if adjusted <= 0.01:
                scores[index] = -np.log10(max(adjusted, np.finfo(float).tiny))
    result["mdic3_lr_official_neglog10_bonferroni_p"] = scores
    result["mdic3_lr_official_raw_correlation"] = raw_correlations
    return result

def read_cellchat_scores(case_output: Path) -> pd.DataFrame | None:
    path = case_output / "lr_score.tsv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, sep="\t", dtype={"sender_type": str, "receiver_type": str})
    if frame.empty:
        return pd.DataFrame(columns=["sender_type", "receiver_type", "ligand", "receptor", "cellchat_probability"])
    frame["ligand"] = frame["ligand"].map(gene_label)
    frame["receptor"] = frame["receptor"].map(gene_label)
    frame = frame.rename(columns={"score": "cellchat_probability"})
    return frame.groupby(["sender_type", "receiver_type", "ligand", "receptor"], as_index=False)["cellchat_probability"].sum()

def _cpdb_gene_columns(frame: pd.DataFrame) -> tuple[str, str]:
    for first, second in (("gene_a", "gene_b"), ("partner_a", "partner_b")):
        if first in frame.columns and second in frame.columns:
            return first, second
    raise ValueError("CellPhoneDB output lacks gene_a/gene_b identities")

def read_cpdb_scores(case_output: Path) -> pd.DataFrame | None:
    mean_paths = sorted((case_output / "cpdb_out").glob("statistical_analysis_means*.txt"))
    if not mean_paths:
        return None
    means = pd.read_csv(mean_paths[-1], sep="\t")
    first, second = _cpdb_gene_columns(means)
    pair_columns = [column for column in means.columns if "|" in str(column)]
    means = means[[first, second, *pair_columns]].copy()
    means["ligand"] = means[first].map(gene_label)
    means["receptor"] = means[second].map(gene_label)
    long = means.melt(
        id_vars=["ligand", "receptor"], value_vars=pair_columns,
        var_name="type_pair", value_name="cellphonedb_mean",
    )
    long[["sender_type", "receiver_type"]] = long["type_pair"].str.split("|", n=1, expand=True)
    long["cellphonedb_mean"] = pd.to_numeric(long["cellphonedb_mean"], errors="coerce").fillna(0)

    p_paths = sorted((case_output / "cpdb_out").glob("statistical_analysis_pvalues*.txt"))
    if p_paths:
        pvalues = pd.read_csv(p_paths[-1], sep="\t")
        p_first, p_second = _cpdb_gene_columns(pvalues)
        pvalues["ligand"] = pvalues[p_first].map(gene_label)
        pvalues["receptor"] = pvalues[p_second].map(gene_label)
        p_long = pvalues.melt(
            id_vars=["ligand", "receptor"], value_vars=[column for column in pair_columns if column in pvalues],
            var_name="type_pair", value_name="cellphonedb_pvalue",
        )
        p_long[["sender_type", "receiver_type"]] = p_long["type_pair"].str.split("|", n=1, expand=True)
        p_long["cellphonedb_pvalue"] = pd.to_numeric(p_long["cellphonedb_pvalue"], errors="coerce").fillna(1)
        long = long.merge(
            p_long[["sender_type", "receiver_type", "ligand", "receptor", "cellphonedb_pvalue"]],
            on=["sender_type", "receiver_type", "ligand", "receptor"], how="left",
        )
        long["cellphonedb_significant_mean"] = long["cellphonedb_mean"] * (long["cellphonedb_pvalue"] < 0.05)
    return long.drop(columns="type_pair").groupby(
        ["sender_type", "receiver_type", "ligand", "receptor"], as_index=False
    ).sum(numeric_only=True)

def merge_optional(table: pd.DataFrame, scores: pd.DataFrame | None) -> pd.DataFrame:
    if scores is None:
        return table
    keys = ["sender_type", "receiver_type", "ligand", "receptor"]
    result = table.merge(scores, on=keys, how="left", validate="one_to_one")
    for column in scores.columns.difference(keys):
        result[column] = result[column].fillna(0)
    return result

def evaluate_scores(table: pd.DataFrame) -> pd.DataFrame:
    excluded = {
        "sender_type", "receiver_type", "ligand", "receptor", "effect",
        "eligible_edge_count", "truth_edge_count", "truth_present",
        "truth_density", "truth_effect_density", "cellphonedb_pvalue",
    }
    score_columns = [
        column for column in table.columns
        if column not in excluded
        and not column.endswith("_coupling_mean")
        and column != "mdic3_lr_official_raw_correlation"
    ]
    y_binary = table["truth_present"].to_numpy(dtype=int)
    y_continuous = table["truth_effect_density"].to_numpy(dtype=float)
    if np.unique(y_binary).size != 2:
        raise ValueError("typepair-LR truth must contain both positive and negative examples")
    rows = []
    for column in score_columns:
        score = pd.to_numeric(table[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(score).all():
            raise ValueError(f"non-finite score in {column}")
        precision, recall, _ = precision_recall_curve(y_binary, score)
        denominator = precision + recall
        f1 = np.divide(
            2 * precision * recall,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        if np.ptp(score) == 0 or np.ptp(y_continuous) == 0:
            spearman = 0.0
            pearson = 0.0
        else:
            spearman = pd.Series(score).corr(pd.Series(y_continuous), method="spearman")
            pearson = pd.Series(score).corr(pd.Series(y_continuous), method="pearson")
        rows.append({
            "method_score": column,
            "auroc": float(roc_auc_score(y_binary, score)),
            "average_precision": float(average_precision_score(y_binary, score)),
            "f1_max": float(f1.max()),
            "spearman_effect_density": float(spearman) if np.isfinite(spearman) else 0.0,
            "pearson_effect_density": float(pearson) if np.isfinite(pearson) else 0.0,
            "ndcg_effect_density": float(ndcg_score(y_continuous[None, :], score[None, :])),
        })
    return pd.DataFrame(rows)

def run_case(args: argparse.Namespace, case_key: str) -> None:
    match = CASE_PATTERN.fullmatch(case_key)
    if not match:
        raise ValueError(f"invalid case key: {case_key}")
    gene_num, dataset_id, _, seed = map(int, match.groups())
    case_dir = args.sim_root / case_key
    truth_dir = args.truth_root / case_key if args.truth_root else None
    metadata = read_metadata(case_dir, truth_dir)
    cells = metadata.index.tolist()
    distance = align_square(read_labeled_matrix(case_dir / "D.txt"), cells, "distance")
    tmenet_case_root = args.tmenet_root / case_key
    if args.tmenet_overlay_root is not None:
        overlay_case_root = args.tmenet_overlay_root / case_key
        if (overlay_case_root / "score.npz").exists() and (overlay_case_root / "metrics.json").exists():
            tmenet_case_root = overlay_case_root
    selected_k = args.k
    if selected_k is None:
        tmenet_metrics = tmenet_case_root / "metrics.json"
        selected_k = int(json.loads(tmenet_metrics.read_text())["spatial_k"])
    candidate_mask = tie_inclusive_knn(distance, cells, selected_k)
    lr_pairs = read_lr_network(case_dir, args.lr_root, gene_num, dataset_id)
    truth, audit = build_truth(case_dir, metadata, lr_pairs, candidate_mask, seed, truth_dir)
    expression = read_expression(case_dir, cells)
    raw_expression = read_raw_expression(case_dir, cells)
    tmenet_c = load_coupling_score(
        tmenet_case_root / "score.npz", cells, "TMENet"
    )
    table, tmenet_audit = add_conserved_coupling_attribution(
        truth, expression, tmenet_c, metadata, candidate_mask, "tmenet"
    )
    attribution_audit = dict(tmenet_audit)
    tmenet_symmetric_c = 0.5 * (tmenet_c + tmenet_c.T)
    table, symmetric_audit = add_conserved_coupling_attribution(
        table, expression, tmenet_symmetric_c, metadata, candidate_mask,
        "tmenet_symmetric",
    )
    attribution_audit.update(symmetric_audit)
    if args.coupling_direction_diagnostics:
        # C is used as a cell-cell coupling strength in the post-hoc LR layer.
        # These pre-specified alternatives diagnose whether its directed part is
        # identifiable; they must not silently replace the primary raw-C score.
        table, transpose_audit = add_conserved_coupling_attribution(
            table, expression, tmenet_c.T, metadata, candidate_mask,
            "tmenet_transpose_diagnostic",
        )
        attribution_audit.update(transpose_audit)
    if args.mdic3_root:
        mdic3_c = load_coupling_score(
            args.mdic3_root / case_key / "score.npz", cells, "MDIC3"
        )
        table, mdic3_audit = add_conserved_coupling_attribution(
            table, expression, mdic3_c, metadata, candidate_mask, "mdic3"
        )
        attribution_audit.update(mdic3_audit)
    table = add_mdic3_lr_official(table, raw_expression, metadata)
    if args.cellchat_root:
        table = merge_optional(table, read_cellchat_scores(args.cellchat_root / case_key))
    if args.cpdb_root:
        table = merge_optional(table, read_cpdb_scores(args.cpdb_root / case_key))
    metrics = evaluate_scores(table)

    output_dir = args.output_root / case_key
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "lr_scores_and_truth.tsv", sep="\t", index=False)
    metrics.to_csv(output_dir / "metrics.tsv", sep="\t", index=False)
    manifest = {
        "case": case_key,
        "direction": "regulator_ligand_sender_to_target_receptor_receiver",
        "evaluation_unit": "sender_cell_type_x_receiver_cell_type_x_lr_pair",
        "candidate_policy": f"directed_tie_inclusive_knn{selected_k}_excluding_self",
        "continuous_truth": "supplied_effect_times_realized_edge_density",
        "primary_tmenet_lr_score": "tmenet_symmetric_lr_conserved_attribution",
        "tmenet_case_source": str(tmenet_case_root),
        "tmenet_lr_definition": (
            "mean(C_ij,C_ji) times directed positive-rank LR support normalized over all supplied LR pairs "
            "within each directed candidate cell edge; mean over candidate edges in the type pair"
        ),
        "raw_directed_c_role": "direction-identifiability_diagnostic_not_primary",
        "mdic3_lr_official_definition": (
            "official MDIC3_LR sorted-expression Pearson test with raw-P filtering, "
            "Bonferroni correction, and -log10 adjusted-P score"
        ),
        "mdic3_c_attribution_role": (
            "diagnostic same-attribution comparator, not an official MDIC3 output"
            if args.mdic3_root else "not_requested"
        ),
        "lr_pair_count": int(len(lr_pairs)),
        "lr_candidate_source": lr_pairs.attrs.get("candidate_source"),
        "contains_false_lr_candidates": lr_pairs.attrs.get("contains_false_lr_candidates", False),
        "input_lr_row_count": int(lr_pairs.attrs.get("input_row_count", len(lr_pairs))),
        "duplicate_lr_row_count": int(lr_pairs.attrs.get("duplicate_lr_row_count", 0)),
        **attribution_audit,
        **audit,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False), flush=True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-root", type=Path, default=Path("data/simulation/simdata"))
    parser.add_argument("--lr-root", type=Path, default=Path("data/simulation/ccc_networks"))
    parser.add_argument("--truth-root", type=Path)
    parser.add_argument("--tmenet-root", type=Path, required=True)
    parser.add_argument("--tmenet-overlay-root", type=Path)
    parser.add_argument("--mdic3-root", type=Path)
    parser.add_argument("--cellchat-root", type=Path)
    parser.add_argument("--cpdb-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-key", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--coupling-direction-diagnostics", action="store_true",
        help=(
            "also score raw directed C and C.T as explicitly labelled diagnostics; "
            "the primary TMENet-LR score remains the symmetric coupling attribution"
        ),
    )
    parser.add_argument(
        "--k", type=int,
        help="fixed K override; by default read spatial_k from each TMENet metrics.json",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.all:
        case_keys = sorted(
            path.relative_to(args.sim_root).as_posix()
            for path in args.sim_root.glob("geneNum_*_id_*/cellType*/seed*")
            if CASE_PATTERN.fullmatch(path.relative_to(args.sim_root).as_posix())
        )
    else:
        case_keys = args.case_key
    if not case_keys:
        raise ValueError("provide --all or at least one --case-key")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard-count/shard-index")
    case_keys = [
        case_key for index, case_key in enumerate(case_keys)
        if index % args.shard_count == args.shard_index
    ]
    for case_key in case_keys:
        print(f"[LR benchmark] {case_key}", flush=True)
        run_case(args, case_key)

if __name__ == "__main__":
    main()
