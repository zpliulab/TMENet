#!/usr/bin/env python3
"""Build label-safe ATAC GRN priors for regenerated scMultiSim cases.

The prior convention is regulator x target.  It is constructed directly from
region-to-gene, mean ATAC accessibility, and region-to-TF mappings.  Unlike the
legacy notebook, directed edges are never symmetrized by an undirected PCA-CMI
filter.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

CASE_PATTERN = re.compile(r"geneNum_([0-9]+)_id_([0-9]+)/cellType([0-9]+)/seed([0-9]+)")

def read_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite matrix: {path}")
    return frame

def atomic_frame(frame: pd.DataFrame, path: Path) -> None:
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", prefix=f".{path.name}.",
        suffix=".tmp", dir=path.parent, delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            frame.to_csv(handle, sep="\t")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

def network_truth(grn_root: Path, gene_num: int, dataset_id: int) -> pd.DataFrame:
    edges = pd.read_csv(grn_root / f"geneNum_{gene_num}_{dataset_id}.txt", sep="\t")
    truth = pd.DataFrame(0, index=map(str, range(1, gene_num + 1)), columns=map(str, range(1, gene_num + 1)))
    for edge in edges.itertuples(index=False):
        truth.loc[str(int(edge.regulator)), str(int(edge.target))] = 1
    return truth

def process_case(case: Path, root: Path, grn_root: Path, force: bool) -> dict[str, object]:
    relative = case.relative_to(root).as_posix()
    match = CASE_PATTERN.fullmatch(relative)
    if not match:
        raise ValueError(f"invalid case path: {relative}")
    gene_num, dataset_id, _, _ = map(int, match.groups())
    complete = case / "PREPROCESS_COMPLETE.json"
    if complete.exists() and not force:
        return json.loads(complete.read_text())
    required = [
        "A.txt", "atac_data.txt", "atac_region2gene.txt", "atac_region2tf.txt",
        "tf_ids.txt", "metadata.txt", "COMPLETE.json",
    ]
    missing = [name for name in required if not (case / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing {missing}")

    a = read_matrix(case / "A.txt")
    atac = read_matrix(case / "atac_data.txt")
    region_to_gene = read_matrix(case / "atac_region2gene.txt")
    region_to_tf = read_matrix(case / "atac_region2tf.txt")
    common_regions = [region for region in atac.index if region in region_to_gene.index and region in region_to_tf.index]
    if len(common_regions) != len(atac.index):
        raise ValueError("ATAC and region mappings do not share the same region set")
    atac = atac.loc[common_regions]
    region_to_gene = region_to_gene.loc[common_regions]
    region_to_tf = region_to_tf.loc[common_regions]
    genes = [str(index) for index in range(1, gene_num + 1)]
    if list(region_to_gene.columns) == [f"V{index}" for index in range(1, gene_num + 1)]:
        region_to_gene.columns = genes
    if set(region_to_gene.columns) != set(genes):
        raise ValueError("region-to-gene columns differ from simulated genes")
    region_to_gene = region_to_gene.loc[:, genes]
    tf_ids = [line.strip() for line in (case / "tf_ids.txt").read_text().splitlines() if line.strip()]
    if tf_ids and tf_ids[0] == "Unnamed: 0":
        tf_ids = tf_ids[1:]
    if list(region_to_tf.columns) == [f"V{index}" for index in range(1, len(tf_ids) + 1)]:
        region_to_tf.columns = tf_ids
    if set(tf_ids) != set(region_to_tf.columns):
        raise ValueError("TF identities differ between tf_ids and region-to-TF")
    region_to_tf = region_to_tf.loc[:, tf_ids]

    mean_accessibility = atac.mean(axis=1)
    target_by_tf = region_to_gene.T.mul(mean_accessibility, axis=1) @ region_to_tf
    regulator_by_target = target_by_tf.T
    full_weighted = pd.DataFrame(0.0, index=genes, columns=genes)
    full_weighted.loc[tf_ids, genes] = regulator_by_target.loc[tf_ids, genes].to_numpy()
    maximum = float(full_weighted.to_numpy().max())
    if maximum > 0:
        full_weighted /= maximum
    full_binary = (full_weighted > 0).astype(int)
    np.fill_diagonal(full_binary.values, 0)
    np.fill_diagonal(full_weighted.values, 0)

    metadata = pd.read_csv(case / "metadata.txt", sep="\t", header=None, names=["cell", "cell_type"], dtype=str)
    cell_types = sorted(metadata["cell_type"].unique())
    labels = [f"{cell_type}_{gene}" for cell_type in cell_types for gene in genes]
    expected_a_rows = labels
    if list(a.index) != expected_a_rows:
        raise ValueError("A row order does not match coarse-cell-type x gene labels")
    block_size = gene_num
    b_values = np.zeros((len(labels), len(labels)), dtype=float)
    weighted_values = full_weighted.to_numpy(dtype=float)
    for block_index in range(len(cell_types)):
        start = block_index * block_size
        b_values[start:start + block_size, start:start + block_size] = weighted_values
    b_star = pd.DataFrame(b_values, index=labels, columns=labels)
    truth = network_truth(grn_root, gene_num, dataset_id)

    atomic_frame(target_by_tf, case / "pre_grn_gene2tf.txt")
    atomic_frame(full_binary, case / "pre_grn_binary_matrix.txt")
    atomic_frame(full_weighted, case / "pre_grn_weighted_matrix.txt")
    atomic_frame(b_star, case / "B_star.txt")
    atomic_frame(truth, case / "gold_GRN.txt")
    report = {
        "case": relative,
        "method": "direct_ATAC_region_gene_T_x_mean_accessibility_x_region_TF",
        "matrix_convention": "regulator_x_target",
        "cell_type_policy": "same_static_ATAC_prior_repeated_on_coarse_type_diagonal_blocks",
        "pca_cmi_removed": True,
        "reason": "legacy PCA-CMI treated a directed prior as an undirected symmetric graph",
        "gene_num": gene_num,
        "tf_count": len(tf_ids),
        "coarse_cell_type_count": len(cell_types),
        "prior_edges": int(full_binary.to_numpy().sum()),
        "prior_density_tf_candidates": float(full_binary.loc[tf_ids].to_numpy().mean()),
        "b_star_shape": list(b_star.shape),
    }
    complete.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--grn-root", type=Path, required=True)
    parser.add_argument("--case-key", action="append", default=[])
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cases = sorted(path.parent for path in args.root.glob("geneNum_*_id_*/cellType*/seed*/COMPLETE.json"))
    if args.case_key:
        requested = set(args.case_key)
        cases = [case for case in cases if case.relative_to(args.root).as_posix() in requested]
    cases = [case for index, case in enumerate(cases) if index % args.shard_count == args.shard_index]
    failures = []
    for position, case in enumerate(cases, start=1):
        print(f"[{position}/{len(cases)}] {case.relative_to(args.root)}", flush=True)
        try:
            report = process_case(case, args.root, args.grn_root, args.force)
            print(f"  prior_edges={report['prior_edges']}", flush=True)
        except Exception as error:
            failures.append({"case": str(case), "error": str(error)})
            print(f"  FAILED: {error}", flush=True)
    if failures:
        (args.root / f"preprocess_failures_{args.shard_index}.json").write_text(json.dumps(failures, indent=2))
        raise SystemExit(1)

if __name__ == "__main__":
    main()
