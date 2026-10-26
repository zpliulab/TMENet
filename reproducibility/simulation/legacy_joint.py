#!/usr/bin/env python3
"""Resume-safe joint TMENet batch saving both GRN B and cell-coupling C."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "benchmarks"))
from grn_benchmark_common import (
    enumerate_cases,
    evaluate_matrix,
    read_expression,
    read_tf_ids,
    read_truth,
    write_json_atomic,
)
from ccc_benchmark_common import evaluate_cell_pair_matrix, load_case_truth_and_candidates
from optimizer_adapter import cpu_moNMTF, gpu_moNMTF
from legacy_io import load_matrix, align_distance_and_knn, build_blockdiag_B_mask

def stratum(gene_num: int) -> str:
    if gene_num <= 40:
        return "small_10_40"
    if gene_num <= 70:
        return "medium_41_70"
    return "large_71_100"

def select_spatial_k(case, mode: str, fixed_k: int):
    if mode == "fixed":
        return fixed_k, {"selection_rule": "fixed", "selected_k": fixed_k}
    coordinates = pd.read_csv(case.data_dir / "coords.csv", index_col=0)
    coordinates.index = coordinates.index.astype(str)
    _, _, cells = load_matrix(case.data_dir / "A.txt")
    cells = [str(value) for value in cells]
    values = coordinates.loc[cells].iloc[:, :2].to_numpy(dtype=float)
    if len(np.unique(values, axis=0)) != len(values):
        raise ValueError(f"duplicate spatial coordinates: {case.key}")
    candidates = set()
    for triangle in Delaunay(values).simplices:
        for first, second in ((0, 1), (0, 2), (1, 2)):
            i, j = sorted((int(triangle[first]), int(triangle[second])))
            candidates.add((i, j))
    degrees = np.zeros(len(values), dtype=int)
    edge_count = 0
    for i, j in candidates:
        midpoint = (values[i] + values[j]) / 2.0
        radius2 = float(np.sum((values[i] - values[j]) ** 2) / 4.0)
        squared = np.sum((values - midpoint) ** 2, axis=1)
        others = np.ones(len(values), dtype=bool)
        others[[i, j]] = False
        tolerance = 1e-10 * max(1.0, radius2)
        if np.any(squared[others] <= radius2 + tolerance):
            continue
        degrees[i] += 1
        degrees[j] += 1
        edge_count += 1
    if edge_count == 0 or np.any(degrees == 0):
        raise ValueError(f"invalid Gabriel graph: {case.key}")
    mean_degree = float(degrees.mean())
    selected_k = max(3, min(12, int(np.rint(mean_degree))))
    return selected_k, {
        "selection_rule": "clip(round(mean_Gabriel_degree),3,12)",
        "selected_k": selected_k,
        "gabriel_mean_degree": mean_degree,
        "gabriel_median_degree": float(np.median(degrees)),
        "gabriel_edge_count": edge_count,
    }

def load_inputs(case, k, c_prior_mode):
    a_raw, _, cells = load_matrix(case.data_dir / "A.txt")
    a_log = np.log1p(a_raw)
    a = (a_log - a_log.min()) / (a_log.max() - a_log.min() + 1e-12)
    b_star, _, _ = load_matrix(case.data_dir / "B_star.txt")
    c_prior_file = (
        "C_star.txt"
        if c_prior_mode == "binary"
        else "C_prior_knn4_weighted.txt"
    )
    c_star, c_rows, c_cols = load_matrix(case.data_dir / c_prior_file)
    cell_labels = [str(value) for value in cells]
    c_frame = pd.DataFrame(
        c_star,
        index=[str(value) for value in c_rows],
        columns=[str(value) for value in c_cols],
    )
    c_star = c_frame.loc[cell_labels, cell_labels].to_numpy(dtype=float)
    d, d_rows, d_cols = load_matrix(case.data_dir / "D.txt")
    d, knn = align_distance_and_knn(d, d_rows, d_cols, cell_labels, k)
    # C_star was generated under the legacy K=4 mask. Projection is required
    # when a geometry-derived K is smaller, otherwise initialization and mask
    # constraints conflict.
    c_star = np.where(knn, c_star, 0.0)
    return a, b_star, c_star, d, knn, cell_labels

def make_options(case, a, b_star, c_star, d, knn, parameters, k):
    rng = np.random.RandomState(42 + case.seed)
    eta = 1e-5 if case.gene_num >= 50 and case.cell_type_count >= 3 else 0.001
    return {
        **parameters,
        "eta": eta,
        "p": 1,
        "display": False,
        "maxIter": int(os.environ.get("TMENET_MAX_ITER", "1000")),
        "minIter": 50,
        "adaptive_step": True,
        "stationarity_tol": 0.02,
        "score_stability_tol": float(os.environ.get("TMENET_SCORE_STABILITY_TOL", "0.001")),
        "score_stability_check_every": int(os.environ.get("TMENET_SCORE_STABILITY_CHECK_EVERY", "25")),
        "score_stability_patience": int(os.environ.get("TMENET_SCORE_STABILITY_PATIENCE", "3")),
        "score_stability_min_iter": int(os.environ.get("TMENET_SCORE_STABILITY_MIN_ITER", "300")),
        "lambda_X": 0.0,
        "use_weighted_ball": True,
        "alpha_in": 0.5,
        "alpha_out": 0.49,
        "zero_diag_C": False,
        "row_normalize_C": False,
        "zero_diag_B": False,
        "B_mask": build_blockdiag_B_mask(case.gene_num, case.cell_type_count),
        "scale_every": 0,
        "K": k,
        "seed": 42 + case.seed,
        "rho_min": 1e-6,
        "rho_max": 1e6,
        "D": d,
        "M_knn": knn,
        "mask_conflict_policy": "error",
        "mask_conflict_tolerance": 0.0,
        "initial_B": rng.uniform(0, 1, size=b_star.shape),
        "initial_C": rng.uniform(0, 1, size=c_star.shape),
        "initial_X": rng.uniform(0, 1, size=a.shape),
        "return_info": True,
    }

def save_score(score: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        score=score.to_numpy(dtype=float),
        genes=np.asarray(score.index, dtype=str),
    )
    temporary.replace(path)

def save_coupling(score: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        score=score.to_numpy(dtype=float),
        cells=np.asarray(score.index, dtype=str),
    )
    temporary.replace(path)

def run_case(
    case, truth_root: Path, output_root: Path, parameters, k_mode, fixed_k,
    c_prior_mode,
):
    selected_k, k_diagnostics = select_spatial_k(case, k_mode, fixed_k)
    a, b_star, c_star, d, knn, input_cells = load_inputs(
        case, selected_k, c_prior_mode
    )
    opts = make_options(case, a, b_star, c_star, d, knn, parameters, selected_k)
    backend_name = "cpu" if case.gene_num <= 40 else "gpu"
    optimizer = cpu_moNMTF if backend_name == "cpu" else gpu_moNMTF
    start = time.perf_counter()
    b_opt, _, c_opt, obj_aug, obj_raw, r_values, s_values, info = optimizer(
        a, b_star, c_star, opts
    )
    elapsed = time.perf_counter() - start

    expression = read_expression(case.data_dir / "rnaseq_data.txt")
    genes = list(expression.index)
    tf_ids = read_tf_ids(case.data_dir / "tf_ids.txt")
    truth = read_truth(
        truth_root / f"geneNum_{case.gene_num}_{case.dataset_id}_adj.txt", genes
    )
    records = []
    case_output = output_root / case.key
    for block in range(case.cell_type_count):
        first = block * case.gene_num
        last = first + case.gene_num
        score = pd.DataFrame(b_opt[first:last, first:last], index=genes, columns=genes)
        save_score(score, case_output / f"cell_type_block={block + 1}" / "score.npz")
        records.append({"cell_type_block": block + 1, **evaluate_matrix(score, truth, tf_ids)})

    cells, ccc_truth, candidate_mask = load_case_truth_and_candidates(
        case.data_dir, k=selected_k
    )
    if input_cells != cells:
        raise ValueError(f"A/C cell order differs from canonical CCC order: {case.key}")
    if c_opt.shape != (len(cells), len(cells)):
        raise ValueError(
            f"optimized C shape {c_opt.shape} differs from cell shape "
            f"{(len(cells), len(cells))}: {case.key}"
        )
    if not np.isfinite(c_opt).all() or c_opt.min() < -1e-8:
        raise ValueError(f"invalid optimized C values: {case.key}")
    c_score = pd.DataFrame(np.maximum(c_opt, 0.0), index=cells, columns=cells)
    save_coupling(c_score, case_output / "score.npz")
    ccc_metrics = evaluate_cell_pair_matrix(c_score, ccc_truth, candidate_mask)
    # C's directed component is not identifiable robustly in the joint
    # factorization.  The manuscript-level estimand is therefore an unordered
    # cell-cell coupling strength, with LR direction assigned only downstream.
    c_undirected = 0.5 * (c_score.to_numpy() + c_score.to_numpy().T)
    truth_values = ccc_truth.to_numpy(dtype=bool)
    truth_undirected = truth_values | truth_values.T
    candidate_undirected = np.triu(candidate_mask | candidate_mask.T, k=1)
    c_undirected_frame = pd.DataFrame(c_undirected, index=cells, columns=cells)
    save_coupling(c_undirected_frame, case_output / "score_undirected.npz")
    ccc_undirected_metrics = evaluate_cell_pair_matrix(
        c_undirected_frame,
        pd.DataFrame(truth_undirected, index=cells, columns=cells),
        candidate_undirected,
    )
    payload = {
        "method": "tmenet_joint_fixed_hybrid",
        "case": case.key,
        "gene_num": case.gene_num,
        "dataset_id": case.dataset_id,
        "cell_type_count": case.cell_type_count,
        "seed": case.seed,
        "stratum": stratum(case.gene_num),
        "backend": backend_name,
        "parameters": parameters,
        "spatial_k_mode": k_mode,
        "spatial_k": selected_k,
        "c_prior_mode": c_prior_mode,
        "spatial_k_diagnostics": k_diagnostics,
        "elapsed_seconds": elapsed,
        "final_raw_objective": float(obj_raw[-1]),
        "final_augmented_objective": float(obj_aug[-1]),
        "final_primal_residual": float(r_values[-1]),
        "final_dual_residual": float(s_values[-1]),
        "optimizer_info": info,
        "candidate_policy": "TF_to_gene_excluding_self",
        "f1_policy": "maximum_on_PR_curve_evaluation_only",
        "cell_types": records,
        "ccc_candidate_policy": (
            f"directed_tie_inclusive_knn{selected_k}_excluding_self"
        ),
        "ccc_truth_definition": "nonzero_gold_CCC",
        "ccc_metrics": ccc_undirected_metrics,
        "ccc_primary_estimand": "undirected_mean_Cij_Cji",
        "ccc_directed_diagnostic_metrics": ccc_metrics,
    }
    for name in ("auroc", "average_precision", "f1_max"):
        payload[f"mean_{name}"] = float(np.mean([record[name] for record in records]))
    return payload

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-root", type=Path, default=Path("data/simulation/simdata_v3"))
    parser.add_argument("--truth-root", type=Path, default=Path("data/simulation/grn_networks"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/simulation_strata.json"))
    parser.add_argument("--case-key", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-gene-num", type=int)
    parser.add_argument("--max-gene-num", type=int)
    parser.add_argument("--cell-type-count", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--manifest-name")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--k-mode", choices=("fixed", "adaptive-gabriel"),
        default="adaptive-gabriel",
    )
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--c-prior-mode", choices=("binary", "weighted"), default="binary",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    if not 1 <= args.k <= 12:
        raise ValueError("--k must be between 1 and 12")
    configs = json.loads(args.config.read_text(encoding="utf-8"))["strata"]
    parameters = {key: value.get("parameters", value.get("best_parameters")) for key, value in configs.items()}
    cases = enumerate_cases(args.sim_root)
    if args.min_gene_num is not None:
        cases = [case for case in cases if case.gene_num >= args.min_gene_num]
    if args.max_gene_num is not None:
        cases = [case for case in cases if case.gene_num <= args.max_gene_num]
    if args.cell_type_count is not None:
        cases = [case for case in cases if case.cell_type_count == args.cell_type_count]
    if args.seed is not None:
        cases = [case for case in cases if case.seed == args.seed]
    if args.case_key:
        wanted = set(args.case_key)
        cases = [case for case in cases if case.key in wanted]
        missing = wanted - {case.key for case in cases}
        if missing:
            raise ValueError(f"unknown cases: {sorted(missing)}")
    if args.limit is not None:
        cases = cases[: args.limit]
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard-count/shard-index")
    filtered_case_count = len(cases)
    cases = [case for index, case in enumerate(cases) if index % args.shard_count == args.shard_index]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_name = args.manifest_name or (
        "manifest.json"
        if args.shard_count == 1
        else f"manifest_worker_{args.shard_index}.json"
    )
    write_json_atomic(
        {
            "method": "tmenet_joint_fixed_hybrid",
            "scope": "joint_GRN_and_cell_coupling_same_optimization_run",
            "sim_root": str(args.sim_root.resolve()),
            "truth_root": str(args.truth_root.resolve()),
            "config": str(args.config.resolve()),
            "case_count": len(cases),
            "filtered_case_count_before_sharding": filtered_case_count,
            "min_gene_num": args.min_gene_num,
            "max_gene_num": args.max_gene_num,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "backend_policy": "cpu_gene_le_40_gpu_gene_gt_40",
            "spatial_k_mode": args.k_mode,
            "fixed_k_when_requested": args.k,
            "adaptive_k_rule": "clip(round(mean_Gabriel_degree),3,12)",
            "c_prior_mode": args.c_prior_mode,
            "saved_outputs": {
                "grn": "case/cell_type_block=N/score.npz",
                "cell_coupling": "case/score.npz",
                "joint_metrics": "case/metrics.json",
            },
            "max_iter": int(os.environ.get("TMENET_MAX_ITER", "1000")),
            "stopping_policy": "stationarity_or_feasible_output_score_stability",
            "score_stability_tol": float(os.environ.get("TMENET_SCORE_STABILITY_TOL", "0.001")),
            "score_stability_check_every": int(os.environ.get("TMENET_SCORE_STABILITY_CHECK_EVERY", "25")),
            "score_stability_patience": int(os.environ.get("TMENET_SCORE_STABILITY_PATIENCE", "3")),
            "score_stability_min_iter": int(os.environ.get("TMENET_SCORE_STABILITY_MIN_ITER", "300")),
            "pid": os.getpid(),
        },
        args.output_root / manifest_name,
    )
    for position, case in enumerate(cases, start=1):
        metric_path = args.output_root / case.key / "metrics.json"
        if metric_path.exists() and not args.force:
            print(f"[{position}/{len(cases)}] skip complete {case.key}", flush=True)
            continue
        print(f"[{position}/{len(cases)}] {case.key} backend={'cpu' if case.gene_num <= 40 else 'gpu'}", flush=True)
        payload = run_case(
            case, args.truth_root, args.output_root,
            parameters[stratum(case.gene_num)],
            args.k_mode, args.k, args.c_prior_mode,
        )
        write_json_atomic(payload, metric_path)

if __name__ == "__main__":
    main()
