#!/usr/bin/env python3
"""Cell-resolved CCC baselines on the canonical simulated datasets.

All methods are evaluated on the same directed tie-inclusive KNN4 candidate
universe as TMENet.  No method reads gold_CCC until the evaluation step.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ccc_benchmark_common import (
    align_square,
    evaluate_cell_pair_matrix,
    load_case_truth_and_candidates,
    read_labeled_matrix,
)
from grn_benchmark_common import enumerate_cases, read_expression, write_json_atomic

METHODS = ("distance", "lr_product", "distance_lr_product", "mdic3")

def distance_scores(case_dir: Path, cells: list[str], candidate_mask: np.ndarray) -> pd.DataFrame:
    distance = align_square(read_labeled_matrix(case_dir / "D.txt"), cells, "distance")
    values = distance.to_numpy(dtype=float)
    candidate_distances = values[candidate_mask & (values > 0)]
    scale = float(np.median(candidate_distances)) if candidate_distances.size else 1.0
    scale = max(scale, np.finfo(float).eps)
    scores = np.exp(-values / scale)
    np.fill_diagonal(scores, 0.0)
    return pd.DataFrame(scores, index=cells, columns=cells)

def read_lr_pairs(case_dir: Path, lr_root: Path, gene_num: int, dataset_id: int) -> pd.DataFrame:
    case_path = case_dir / "LR_candidate_universe.tsv"
    csv_path = lr_root / f"lr_pairs_{gene_num}_{dataset_id}.csv"
    tsv_path = lr_root / f"geneNum_{gene_num}_{dataset_id}.txt"
    if case_path.exists():
        pairs = pd.read_csv(case_path, sep="\t")
    elif csv_path.exists():
        pairs = pd.read_csv(csv_path)
    elif tsv_path.exists():
        pairs = pd.read_csv(tsv_path, sep="\t")
    else:
        raise FileNotFoundError(f"missing simulated LR pairs for geneNum={gene_num}, id={dataset_id}")
    pairs.columns = [str(column).strip().lower() for column in pairs.columns]
    if "regulator" in pairs.columns:
        pairs = pairs.rename(columns={"regulator": "ligand"})
    if "target" in pairs.columns:
        pairs = pairs.rename(columns={"target": "receptor"})
    if not {"ligand", "receptor"}.issubset(pairs.columns):
        raise ValueError("LR table must contain ligand/receptor or regulator/target")
    pairs = pairs[["ligand", "receptor"]].astype(str).drop_duplicates()
    return pairs

def lr_product_scores(
    case,
    cells: list[str],
    lr_root: Path,
) -> tuple[pd.DataFrame, int]:
    expression = read_expression(case.data_dir / "rnaseq_data.txt").loc[:, cells]
    transformed = np.log1p(expression.to_numpy(dtype=float))
    row_minimum = transformed.min(axis=1, keepdims=True)
    row_range = transformed.max(axis=1, keepdims=True) - row_minimum
    normalized = np.divide(
        transformed - row_minimum,
        row_range,
        out=np.zeros_like(transformed),
        where=row_range > 0,
    )
    positions = {gene: index for index, gene in enumerate(expression.index)}
    pairs = read_lr_pairs(case.data_dir, lr_root, case.gene_num, case.dataset_id)
    valid_pairs = [
        (ligand, receptor)
        for ligand, receptor in pairs.itertuples(index=False, name=None)
        if ligand in positions and receptor in positions
    ]
    if not valid_pairs:
        raise ValueError(f"no simulated LR pairs overlap expression genes: {case.key}")
    scores = np.zeros((len(cells), len(cells)), dtype=float)
    for ligand, receptor in valid_pairs:
        scores += np.outer(normalized[positions[ligand]], normalized[positions[receptor]])
    scores /= len(valid_pairs)
    np.fill_diagonal(scores, 0.0)
    return pd.DataFrame(scores, index=cells, columns=cells), len(valid_pairs)

def mdic3_scores(case, cells: list[str]) -> pd.DataFrame:
    expression = read_expression(case.data_dir / "rnaseq_data.txt").loc[:, cells]
    genes = list(expression.index)
    grn = align_square(
        read_labeled_matrix(case.data_dir / "pre_grn_binary_matrix.txt"),
        genes,
        "pre_grn_binary_matrix",
    ).to_numpy(dtype=float)
    abundance = expression.to_numpy(dtype=float)

    singular_values = np.linalg.svd(abundance, compute_uv=False)
    sigma = np.diag(singular_values)
    if abundance.shape[0] < abundance.shape[1]:
        sigma = np.column_stack((sigma, np.zeros((abundance.shape[0], abundance.shape[1] - abundance.shape[0]))))
    elif abundance.shape[0] > abundance.shape[1]:
        sigma = np.row_stack((sigma, np.zeros((abundance.shape[0] - abundance.shape[1], abundance.shape[1]))))
    mediator = np.linalg.pinv(grn @ sigma) @ abundance

    scores = np.zeros_like(mediator, dtype=float)
    positive = mediator >= 0
    scores[positive] += mediator[positive]
    negative_rows, negative_columns = np.where(~positive)
    scores[negative_columns, negative_rows] += np.abs(mediator[negative_rows, negative_columns])
    if not np.isfinite(scores).all():
        raise ValueError(f"MDIC3 produced non-finite scores: {case.key}")
    np.fill_diagonal(scores, 0.0)
    return pd.DataFrame(scores, index=cells, columns=cells)

def save_score(score: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        score=score.to_numpy(dtype=float),
        cells=np.asarray(score.index, dtype=str),
    )
    temporary.replace(path)

def run_case(case, method: str, lr_root: Path, output_root: Path, k: int) -> dict[str, object]:
    cells, truth, candidate_mask = load_case_truth_and_candidates(case.data_dir, k=k)
    started = time.perf_counter()
    extra: dict[str, object] = {}
    if method == "distance":
        score = distance_scores(case.data_dir, cells, candidate_mask)
        input_definition = "physical distance only"
    elif method == "lr_product":
        score, valid_pair_count = lr_product_scores(case, cells, lr_root)
        extra["valid_lr_pair_count"] = valid_pair_count
        input_definition = "RNA expression plus simulated LR identities; no GRN, CCC prior, or gold"
    elif method == "distance_lr_product":
        lr_score, valid_pair_count = lr_product_scores(case, cells, lr_root)
        spatial_score = distance_scores(case.data_dir, cells, candidate_mask)
        score = lr_score * spatial_score
        extra["valid_lr_pair_count"] = valid_pair_count
        input_definition = (
            "physical distance plus RNA expression and simulated LR identities; "
            "no GRN, CCC prior, or gold"
        )
    elif method == "mdic3":
        score = mdic3_scores(case, cells)
        input_definition = "RNA expression plus pre_grn_binary_matrix; no true GRN or gold CCC"
    else:
        raise ValueError(f"unsupported method: {method}")
    elapsed_seconds = time.perf_counter() - started

    metrics = evaluate_cell_pair_matrix(score, truth, candidate_mask)
    case_output = output_root / method / case.key
    save_score(score, case_output / "score.npz")
    return {
        "method": method,
        "case": case.key,
        "gene_num": case.gene_num,
        "dataset_id": case.dataset_id,
        "cell_type_count": case.cell_type_count,
        "seed": case.seed,
        "input_definition": input_definition,
        "elapsed_seconds": elapsed_seconds,
        "candidate_policy": f"directed_tie_inclusive_knn{k}_excluding_self",
        "truth_definition": "nonzero_gold_CCC",
        "f1_policy": "maximum_on_PR_curve_evaluation_only",
        **extra,
        **metrics,
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--sim-root", type=Path, default=Path("data/simulation/simdata"))
    parser.add_argument("--lr-root", type=Path, default=Path("data/simulation/ccc_networks"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-key", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    cases = enumerate_cases(args.sim_root)
    if args.case_key:
        requested = set(args.case_key)
        cases = [case for case in cases if case.key in requested]
        missing = requested - {case.key for case in cases}
        if missing:
            raise ValueError(f"unknown cases: {sorted(missing)}")
    if args.limit is not None:
        cases = cases[: args.limit]
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard-count/shard-index")
    cases = [
        case for index, case in enumerate(cases)
        if index % args.shard_count == args.shard_index
    ]

    method_root = args.output_root / args.method
    method_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        {
            "method": args.method,
            "scope": "simulated cell-pair CCC",
            "sim_root": str(args.sim_root.resolve()),
            "lr_root": str(args.lr_root.resolve()),
            "case_count": len(cases),
            "candidate_policy": f"directed_tie_inclusive_knn{args.k}_excluding_self",
        },
        method_root / (
            "manifest.json"
            if args.shard_count == 1
            else f"manifest_worker_{args.shard_index}.json"
        ),
    )
    for position, case in enumerate(cases, start=1):
        metrics_path = method_root / case.key / "metrics.json"
        if metrics_path.exists() and not args.force:
            print(f"[{position}/{len(cases)}] skip complete {case.key}", flush=True)
            continue
        print(f"[{position}/{len(cases)}] {args.method} {case.key}", flush=True)
        write_json_atomic(
            run_case(case, args.method, args.lr_root, args.output_root, args.k),
            metrics_path,
        )

if __name__ == "__main__":
    main()
