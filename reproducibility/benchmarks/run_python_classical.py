#!/usr/bin/env python3
"""Run PCC or MI on the canonical simulated-GRN benchmark manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from grn_benchmark_common import (
    Case,
    enumerate_cases,
    evaluate_matrix,
    read_expression,
    read_metadata,
    read_tf_ids,
    read_truth,
    write_json_atomic,
)

def pcc_scores(expression: pd.DataFrame) -> pd.DataFrame:
    scores = expression.T.corr(method="pearson").abs().fillna(0.0)
    np.fill_diagonal(scores.values, 0.0)
    return scores

def mi_scores(expression: pd.DataFrame) -> pd.DataFrame:
    """Estimate pairwise MI without depending on arbitrary gene ordering.

    sklearn's estimator treats one variable as the response.  We therefore
    compute every response direction and average the two estimates.
    """
    cells_by_genes = expression.T.to_numpy(dtype=float)
    genes = list(expression.index)
    n_genes = len(genes)
    directed = np.zeros((n_genes, n_genes), dtype=float)
    for target in range(n_genes):
        feature_pos = np.arange(n_genes) != target
        if np.std(cells_by_genes[:, target]) <= 1e-12:
            continue
        usable = feature_pos & (np.std(cells_by_genes, axis=0) > 1e-12)
        if not usable.any():
            continue
        directed[usable, target] = mutual_info_regression(
            cells_by_genes[:, usable],
            cells_by_genes[:, target],
            discrete_features=False,
            random_state=0,
        )
    scores = 0.5 * (directed + directed.T)
    np.fill_diagonal(scores, 0.0)
    return pd.DataFrame(scores, index=genes, columns=genes)

METHODS = {"pcc": pcc_scores, "mi": mi_scores}

def save_score(score: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        score=score.to_numpy(dtype=float),
        genes=np.asarray(score.index, dtype=str),
    )
    temporary.replace(output_path)

def run_case(case: Case, truth_root: Path, method: str, output_root: Path) -> dict[str, object]:
    expression = read_expression(case.data_dir / "rnaseq_data.txt")
    metadata = read_metadata(case.data_dir / "metadata.txt", expression.columns)
    tf_ids = read_tf_ids(case.data_dir / "tf_ids.txt")
    truth = read_truth(
        truth_root / f"geneNum_{case.gene_num}_{case.dataset_id}_adj.txt",
        expression.index,
    )
    observed_types = list(dict.fromkeys(metadata["cell_type"].tolist()))
    if len(observed_types) != case.cell_type_count:
        raise ValueError(
            f"{case.key}: expected {case.cell_type_count} cell types, found {len(observed_types)}"
        )

    cell_type_records = []
    for cell_type in observed_types:
        cells = metadata.index[metadata["cell_type"] == cell_type]
        if len(cells) < 5:
            raise ValueError(f"{case.key}/{cell_type}: fewer than five cells")
        score = METHODS[method](expression.loc[:, cells])
        metrics = evaluate_matrix(score, truth, tf_ids)
        score_path = output_root / method / case.key / f"cell_type={cell_type}" / "score.npz"
        save_score(score, score_path)
        cell_type_records.append({"cell_type": cell_type, **metrics})

    metric_names = ("auroc", "average_precision", "f1_max")
    mean_metrics = {
        f"mean_{name}": float(np.mean([record[name] for record in cell_type_records]))
        for name in metric_names
    }
    return {
        "method": method,
        "case": case.key,
        "gene_num": case.gene_num,
        "dataset_id": case.dataset_id,
        "cell_type_count": case.cell_type_count,
        "seed": case.seed,
        "candidate_policy": "TF_to_gene_excluding_self",
        "f1_policy": "maximum_on_PR_curve_evaluation_only",
        "cell_types": cell_type_records,
        **mean_metrics,
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--sim-root", type=Path, default=Path("data/simulation/simdata"))
    parser.add_argument("--truth-root", type=Path, default=Path("data/simulation/grn_networks"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-key", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    cases = enumerate_cases(args.sim_root)
    if args.case_key:
        wanted = set(args.case_key)
        cases = [case for case in cases if case.key in wanted]
        missing = wanted - {case.key for case in cases}
        if missing:
            raise ValueError(f"unknown case keys: {sorted(missing)}")
    if args.limit is not None:
        cases = cases[: args.limit]

    method_root = args.output_root / args.method
    method_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": args.method,
        "sim_root": str(args.sim_root.resolve()),
        "truth_root": str(args.truth_root.resolve()),
        "case_count": len(cases),
        "pid": os.getpid(),
    }
    write_json_atomic(manifest, method_root / "manifest.json")

    for position, case in enumerate(cases, start=1):
        result_path = method_root / case.key / "metrics.json"
        if result_path.exists() and not args.force:
            print(f"[{position}/{len(cases)}] skip complete {case.key}", flush=True)
            continue
        print(f"[{position}/{len(cases)}] {args.method} {case.key}", flush=True)
        record = run_case(case, args.truth_root, args.method, args.output_root)
        write_json_atomic(record, result_path)

if __name__ == "__main__":
    main()
