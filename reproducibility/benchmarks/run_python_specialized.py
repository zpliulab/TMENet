#!/usr/bin/env python3
"""Canonical per-cell-type Dictys and SpaGRN simulation runners."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from grn_benchmark_common import (
    enumerate_cases,
    read_expression,
    read_metadata,
    read_tf_ids,
    write_json_atomic,
)

def complete_score(genes: list[str], edges: pd.DataFrame) -> pd.DataFrame:
    score = pd.DataFrame(0.0, index=genes, columns=genes)
    if edges.empty:
        return score
    required = {"regulator", "target", "importance"}
    if set(edges.columns) != required:
        raise ValueError(f"edge columns must be {sorted(required)}")
    edges = edges.copy()
    edges["regulator"] = edges["regulator"].astype(str)
    edges["target"] = edges["target"].astype(str)
    # SpaGRN can emit NaN or infinite importances for a small subset of
    # simulated inputs.  Such values have no valid rank interpretation, so
    # discard only those individual edges; all unreported edges remain zero.
    edges["importance"] = pd.to_numeric(edges["importance"], errors="coerce")
    finite_importance = np.isfinite(edges["importance"].to_numpy(dtype=float))
    edges = edges.loc[finite_importance].copy()
    edges["importance"] = edges["importance"].abs()
    edges = edges[edges["regulator"].isin(genes) & edges["target"].isin(genes)]
    grouped = edges.groupby(["regulator", "target"], sort=False)["importance"].max()
    for (regulator, target), value in grouped.items():
        score.loc[regulator, target] = value
    np.fill_diagonal(score.values, 0.0)
    if not np.isfinite(score.values).all():
        raise ValueError("non-finite inferred scores")
    return score

def run_dictys(expression: pd.DataFrame, prior: pd.DataFrame, tf_ids: list[str]) -> pd.DataFrame:
    from dictys.network import reconstruct

    genes = list(expression.index)
    expression = expression.copy()
    prior = prior.loc[genes, genes].copy()
    np.fill_diagonal(prior.values, 0.0)
    prefix = "g_"
    expression.index = [prefix + gene for gene in genes]
    prior.index = [prefix + gene for gene in genes]
    prior.columns = [prefix + gene for gene in genes]

    with tempfile.TemporaryDirectory(prefix="tmenet_dictys_") as tmp:
        tmp = Path(tmp)
        expression_path = tmp / "expression.tsv"
        prior_path = tmp / "prior.tsv"
        expression.to_csv(expression_path, sep="\t")
        prior.to_csv(prior_path, sep="\t")
        kwargs = {
            "fi_exp": str(expression_path),
            "fi_mask": str(prior_path),
            "fo_weight": str(tmp / "weight.tsv"),
            "fo_loss": str(tmp / "loss.tsv"),
            "fo_stats": str(tmp / "stats.tsv"),
            "fo_meanvar": str(tmp / "meanvar.tsv"),
            "fo_covfactor": str(tmp / "covfactor.tsv"),
            "rseed": 12345,
        }
        nstep = os.environ.get("GRN_DICTYS_NSTEP")
        if nstep:
            kwargs["nstep"] = int(nstep)
        reconstruct(**kwargs)
        weights = pd.read_csv(tmp / "weight.tsv", sep="\t", index_col=0)

    weights.index = [str(value).removeprefix(prefix) for value in weights.index]
    weights.columns = [str(value).removeprefix(prefix) for value in weights.columns]
    valid_tfs = [tf for tf in tf_ids if tf in weights.index]
    edge_rows = []
    for regulator in valid_tfs:
        for target in genes:
            if target in weights.columns:
                edge_rows.append((regulator, target, abs(float(weights.loc[regulator, target]))))
    return complete_score(genes, pd.DataFrame(edge_rows, columns=["regulator", "target", "importance"]))

def run_spagrn(
    expression: pd.DataFrame,
    coordinates: pd.DataFrame,
    tf_ids: list[str],
) -> pd.DataFrame:
    import anndata as ad
    from spagrn.regulatory_network import InferNetwork

    genes = list(expression.index)
    cells = list(expression.columns)
    coordinates = coordinates.loc[cells, ["x", "y"]]
    adata = ad.AnnData(X=expression.T.to_numpy(dtype=float))
    adata.var_names = genes
    adata.obs_names = cells
    adata.obsm["spatial"] = coordinates.to_numpy(dtype=float)
    network = InferNetwork(adata, project_name="tmenet_simulation")
    edges = network.spg(
        data=adata,
        gene_list=genes,
        layer_key=None,
        model="danb",
        latent_obsm_key="spatial",
        n_neighbors=4,
        tf_list=tf_ids,
        save_tmp=False,
        jobs=1,
        cache=False,
        mode="moran",
        fn=None,
    )
    if edges is None:
        raise RuntimeError("SpaGRN returned no adjacency table")
    edges = edges.iloc[:, :3].copy()
    edges.columns = ["regulator", "target", "importance"]
    return complete_score(genes, edges)

def save_score(score: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tsv.tmp")
    score.to_csv(temporary, sep="\t")
    temporary.replace(path)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("dictys", "spagrn"), required=True)
    parser.add_argument("--sim-root", type=Path, default=Path("data/simulation/simdata"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    cases = enumerate_cases(args.sim_root)
    if args.limit is not None:
        cases = cases[: args.limit]
    method_root = args.output_root / args.method
    method_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        {"method": args.method, "sim_root": str(args.sim_root.resolve()), "case_count": len(cases), "pid": os.getpid()},
        method_root / "manifest.json",
    )

    for position, case in enumerate(cases, start=1):
        case_root = method_root / case.key
        done_path = case_root / "INFERENCE_COMPLETE"
        if done_path.exists() and not args.force:
            print(f"[{position}/{len(cases)}] skip complete {case.key}", flush=True)
            continue
        print(f"[{position}/{len(cases)}] {args.method} {case.key}", flush=True)
        expression = read_expression(case.data_dir / "rnaseq_data.txt")
        metadata = read_metadata(case.data_dir / "metadata.txt", expression.columns)
        tf_ids = read_tf_ids(case.data_dir / "tf_ids.txt")
        observed_types = list(dict.fromkeys(metadata["cell_type"].tolist()))
        if len(observed_types) != case.cell_type_count:
            raise ValueError(f"cell-type count mismatch: {case.key}")

        if args.method == "dictys":
            prior = pd.read_csv(case.data_dir / "pre_grn_binary_matrix.txt", sep="\t", index_col=0)
            prior.index = prior.index.astype(str)
            prior.columns = prior.columns.astype(str)
        else:
            coordinates = pd.read_csv(case.data_dir / "coords.csv", index_col=0)
            coordinates.index = coordinates.index.astype(str)
            if set(coordinates.index) != set(expression.columns):
                raise ValueError(f"coordinate/expression cell mismatch: {case.key}")

        for cell_type in observed_types:
            cells = metadata.index[metadata["cell_type"] == cell_type].tolist()
            subset = expression.loc[:, cells]
            if args.method == "dictys":
                score = run_dictys(subset, prior, tf_ids)
            else:
                score = run_spagrn(subset, coordinates, tf_ids)
            save_score(score, case_root / f"cell_type={cell_type}" / "score.tsv")
        done_path.write_text("ok\n", encoding="utf-8")

if __name__ == "__main__":
    main()
