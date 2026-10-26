#!/usr/bin/env python3
"""Prepare TMENet simulation gold_CCC.txt and C_star.txt without truth leakage.

Inputs:
  A.txt                           canonical cell order
  D.txt                           labelled spatial distance matrix
  C_prior_knn4_binary.txt         independently constructed CCC prior
  ../CCI/seed*_cci.txt            scMultiSim ground-truth communication edges

Outputs are only written with --write.  C_star is the prior reordered to A's
cell order; gold_CCC is reconstructed independently from the CCI edge list.
Gold edges are never inserted into C_star.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

def read_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", index_col=0).fillna(0)
    frame.index = frame.index.map(str)
    frame.columns = frame.columns.map(str)
    return frame

def validate_square_labels(frame: pd.DataFrame, name: str, cells: list[str]) -> None:
    if frame.shape[0] != frame.shape[1]:
        raise ValueError(f"{name} is not square: {frame.shape}")
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError(f"{name} contains duplicate labels")
    if set(frame.index) != set(frame.columns) or set(frame.index) != set(cells):
        raise ValueError(f"{name} label set differs from A")

def tie_inclusive_knn(distance: pd.DataFrame, cells: list[str], k: int) -> np.ndarray:
    validate_square_labels(distance, "D", cells)
    values = distance.loc[cells, cells].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("D must contain finite non-negative distances")
    safe = values.copy()
    np.fill_diagonal(safe, np.inf)
    k_eff = min(k, len(cells) - 1)
    if k_eff < 1:
        raise ValueError("At least two cells are required")
    kth = np.sort(safe, axis=1)[:, k_eff - 1]
    mask = safe <= kth[:, None]
    np.fill_diagonal(mask, False)
    return mask

def build_gold(cci_path: Path, cells: list[str]) -> np.ndarray:
    cci = pd.read_csv(cci_path, sep="\t")
    if not {"sender", "receiver"}.issubset(cci.columns):
        raise ValueError("CCI table must contain sender and receiver columns")
    positions = {cell: i for i, cell in enumerate(cells)}
    gold = np.zeros((len(cells), len(cells)), dtype=np.int8)
    unknown = []
    for sender, receiver in zip(cci["sender"], cci["receiver"]):
        sender, receiver = str(sender), str(receiver)
        if sender not in positions or receiver not in positions:
            unknown.append((sender, receiver))
            continue
        gold[positions[sender], positions[receiver]] = 1
    if unknown:
        raise ValueError(f"CCI contains {len(unknown)} edges with unknown cell labels")
    return gold

def atomic_write_matrix(values: np.ndarray, cells: list[str], path: Path) -> None:
    frame = pd.DataFrame(values, index=cells, columns=cells)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", suffix=".tmp",
        prefix=f".{path.name}.", dir=path.parent, delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            frame.to_csv(handle, sep="\t")
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

def process_case(case: Path, k: int, write: bool) -> dict:
    required = (case / "A.txt", case / "D.txt", case / "C_prior_knn4_binary.txt")
    missing = [str(path.name) for path in required if not path.exists()]
    cci_path = case.parent / "CCI" / f"{case.name}_cci.txt"
    if not cci_path.exists():
        missing.append(str(cci_path))
    if missing:
        return {"case": str(case), "status": "missing", "missing": missing}
    try:
        a = read_matrix(case / "A.txt")
        cells = list(a.columns)
        if len(cells) != len(set(cells)):
            raise ValueError("A contains duplicate cell columns")
        distance = read_matrix(case / "D.txt")
        prior = read_matrix(case / "C_prior_knn4_binary.txt")
        validate_square_labels(prior, "C_prior", cells)
        prior_values = prior.loc[cells, cells].to_numpy(dtype=float)
        prior_support = prior_values != 0
        gold = build_gold(cci_path, cells)
        gold_support = gold != 0
        mask = tie_inclusive_knn(distance, cells, k)
        prior_outside = int(np.count_nonzero(prior_support & ~mask))
        gold_outside = int(np.count_nonzero(gold_support & ~mask))
        if prior_outside or gold_outside:
            raise ValueError(
                f"spatial support violation: prior={prior_outside}, gold={gold_outside}"
            )
        if np.count_nonzero(np.diag(prior_support)) or np.count_nonzero(np.diag(gold_support)):
            raise ValueError("self-loop detected in prior or gold")
        if write:
            atomic_write_matrix(gold, cells, case / "gold_CCC.txt")
            atomic_write_matrix(prior_values, cells, case / "C_star.txt")
        return {
            "case": str(case),
            "status": "ok",
            "written": write,
            "cells": len(cells),
            "prior_edges": int(prior_support.sum()),
            "gold_edges": int(gold_support.sum()),
            "gold_edges_missing_from_prior": int(np.count_nonzero(gold_support & ~prior_support)),
            "tie_knn_edges": int(mask.sum()),
        }
    except Exception as exc:
        return {"case": str(case), "status": "error", "error": str(exc)}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--case", type=Path, action="append")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    cases = args.case or sorted(
        path.parent for path in args.root.glob("geneNum_*_id_*/cellType*/seed*/A.txt")
    )
    rows = [process_case(case, args.k, args.write) for case in cases]
    summary = {
        "root": str(args.root),
        "k": args.k,
        "write": args.write,
        "cases": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "total_gold_edges_missing_from_prior": sum(
            row.get("gold_edges_missing_from_prior", 0) for row in rows
        ),
        "max_gold_edges_missing_from_prior": max(
            (row.get("gold_edges_missing_from_prior", 0) for row in rows), default=0
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if any(row["status"] != "ok" for row in rows):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
