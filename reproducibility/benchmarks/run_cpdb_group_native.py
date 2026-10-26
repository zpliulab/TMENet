#!/usr/bin/env python3
"""Run CellPhoneDB and save native directed cell-type-pair scores."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Case:
    key: str
    data_dir: Path
    gene_num: int
    dataset_id: int
    cell_type_count: int
    seed: int

def enumerate_cases(sim_root: Path) -> list[Case]:
    cases = []
    for path in sim_root.glob("geneNum_*_id_*/cellType*/seed*"):
        match = re.fullmatch(
            r"geneNum_([0-9]+)_id_([0-9]+)/cellType([0-9]+)/seed(12|22|32|42|52)",
            path.relative_to(sim_root).as_posix(),
        )
        if match:
            cases.append(
                Case(
                    key=path.relative_to(sim_root).as_posix(),
                    data_dir=path,
                    gene_num=int(match.group(1)),
                    dataset_id=int(match.group(2)),
                    cell_type_count=int(match.group(3)),
                    seed=int(match.group(4)),
                )
            )
    return sorted(cases, key=lambda case: (case.gene_num, case.dataset_id, case.cell_type_count, case.seed))

def prepare_inputs(
    case: Case, case_output: Path, metadata_root: Path | None
) -> tuple[Path, Path, str]:
    expression = pd.read_csv(case.data_dir / "rnaseq_data.txt", sep="\t", index_col=0)
    expression.index = [f"g{int(float(value))}" for value in expression.index]
    metadata = pd.read_csv(
        case.data_dir / "metadata.txt", sep="\t", header=None, names=["original_cell", "cell_type"], dtype=str
    )
    fine_path = case.data_dir / "metadata_fine.tsv"
    resolution = "legacy_coarse_label"
    if fine_path.exists() or metadata_root is not None:
        if not fine_path.exists():
            fine_path = metadata_root / case.key / "metadata_fine.tsv"
            if not fine_path.exists():
                fine_path = metadata_root / case.key / "cell_metadata.tsv"
        fine = pd.read_csv(fine_path, sep="\t", dtype=str)
        if not {"cell", "cell_type"}.issubset(fine.columns):
            raise ValueError(f"fine metadata lacks cell/cell_type: {case.key}")
        metadata = fine[["cell", "cell_type"]].rename(columns={"cell": "original_cell"})
        resolution = (
            "scMultiSim_exact_fine_state"
            if fine_path == case.data_dir / "metadata_fine.tsv"
            else "external_fine_state"
        )
    original_cells = metadata["original_cell"].tolist()
    if set(expression.columns.astype(str)) != set(original_cells):
        raise ValueError(f"expression and metadata cell labels differ: {case.key}")
    expression.columns = expression.columns.astype(str)
    expression = expression.loc[:, original_cells]
    cpdb_cells = [f"C_{index}" for index in range(len(original_cells))]
    expression.columns = cpdb_cells
    cpdb_metadata = pd.DataFrame({"Cell": cpdb_cells, "cell_type": metadata["cell_type"].tolist()})

    input_dir = case_output / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    counts_path = input_dir / "counts.txt"
    metadata_path = input_dir / "meta.txt"
    expression.to_csv(counts_path, sep="\t")
    cpdb_metadata.to_csv(metadata_path, sep="\t", index=False)
    return counts_path, metadata_path, resolution

def group_matrix(means: pd.DataFrame, cell_types: list[str]) -> pd.DataFrame:
    levels = sorted(set(cell_types))
    score = pd.DataFrame(0.0, index=levels, columns=levels)
    for column in means.columns:
        if "|" not in str(column):
            continue
        sender, receiver = str(column).split("|", 1)
        if sender in score.index and receiver in score.columns:
            score.loc[sender, receiver] = float(pd.to_numeric(means[column], errors="coerce").fillna(0).sum())
    return score

def run_case(
    case: Case, db_root: Path, output_root: Path, metadata_root: Path | None,
    iterations: int, threads: int,
) -> None:
    from cellphonedb.src.core.methods import cpdb_statistical_analysis_method
    case_output = output_root / case.key
    score_path = case_output / "group_score.tsv"
    if score_path.exists():
        return
    database_matches = sorted((db_root / f"geneNum_{case.gene_num}_id_{case.dataset_id}").glob("cellphonedb_*.zip"))
    if not database_matches:
        raise FileNotFoundError(f"custom database missing: {case.key}")
    counts_path, metadata_path, metadata_resolution = prepare_inputs(
        case, case_output, metadata_root
    )
    raw_output = case_output / "cpdb_out"
    raw_output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = cpdb_statistical_analysis_method.call(
        cpdb_file_path=str(database_matches[-1]),
        meta_file_path=str(metadata_path),
        counts_file_path=str(counts_path),
        counts_data="gene_name",
        output_path=str(raw_output),
        iterations=iterations,
        threshold=0,
        threads=threads,
        debug_seed=case.seed,
        output_suffix="native",
    )
    means = result.get("means")
    if means is None:
        matches = sorted(raw_output.glob("statistical_analysis_means*.txt"))
        if not matches:
            raise ValueError(f"CellPhoneDB returned no means: {case.key}")
        means = pd.read_csv(matches[-1], sep="\t")
    metadata = pd.read_csv(metadata_path, sep="\t", dtype=str)
    score = group_matrix(means, metadata["cell_type"].tolist())
    score.to_csv(score_path, sep="\t")
    manifest = {
        "method": "CellPhoneDB",
        "case": case.key,
        "resolution": "directed_cell_type_pair",
        "score_definition": "sum_of_mean_expression_over_exact_simulated_LR_pairs",
        "database_policy": "exact_simulated_gene_and_LR_identity_no_human_remapping",
        "cell_type_resolution": metadata_resolution,
        "iterations": iterations,
        "threads": threads,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (case_output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-root", type=Path, default=Path("data/simulation/simdata"))
    parser.add_argument("--db-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--case-key", action="append", default=[])
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    cases = enumerate_cases(args.sim_root)
    if args.case_key:
        requested = set(args.case_key)
        cases = [case for case in cases if case.key in requested]
    cases = [case for index, case in enumerate(cases) if index % args.shard_count == args.shard_index]
    for position, case in enumerate(cases, start=1):
        print(f"[{position}/{len(cases)}] {case.key}", flush=True)
        try:
            run_case(case, args.db_root, args.output_root, args.metadata_root, args.iterations, args.threads)
        except Exception as error:
            failure_dir = args.output_root / case.key
            failure_dir.mkdir(parents=True, exist_ok=True)
            (failure_dir / "failure.json").write_text(
                json.dumps({"case": case.key, "message": str(error)}, indent=2), encoding="utf-8"
            )
            print(f"  FAILED: {error}", flush=True)

if __name__ == "__main__":
    main()
