#!/usr/bin/env python3
"""Design deterministic simulated LR networks independently of intracellular GRNs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PATTERN = re.compile(r"geneNum_([0-9]+)_([0-9]+)\.txt")

def stable_seed(base_seed: int, gene_num: int, dataset_id: int) -> int:
    digest = hashlib.sha256(f"lr-v2:{base_seed}:{gene_num}:{dataset_id}".encode()).digest()
    return int.from_bytes(digest[:4], "little")

def choose_pool_sizes(gene_num: int, edge_count: int) -> tuple[int, int]:
    # Use disjoint role pools covering roughly 30% of genes each. Small pools
    # are capped by the number of true edges so every role gene can be used.
    ligand_count = min(edge_count, max(2, int(round(gene_num * 0.30))))
    receptor_count = min(edge_count, max(2, int(round(gene_num * 0.30))))
    while ligand_count * receptor_count < edge_count:
        if ligand_count <= receptor_count and ligand_count + receptor_count < gene_num:
            ligand_count += 1
        elif ligand_count + receptor_count < gene_num:
            receptor_count += 1
        else:
            raise ValueError(f"cannot place {edge_count} LR edges among {gene_num} genes")
    if ligand_count + receptor_count > gene_num:
        raise ValueError("ligand and receptor pools cannot be disjoint")
    return ligand_count, receptor_count

def sample_true_pairs(
    ligands: np.ndarray, receptors: np.ndarray, edge_count: int, rng: np.random.Generator
) -> pd.DataFrame:
    all_pairs = [(int(ligand), int(receptor)) for ligand in ligands for receptor in receptors]
    chosen: list[tuple[int, int]] = []
    chosen_set: set[tuple[int, int]] = set()
    # A randomized cyclic matching covers both role pools before extra edges
    # are sampled, whenever the requested true-edge count permits it.
    shuffled_ligands = rng.permutation(ligands)
    shuffled_receptors = rng.permutation(receptors)
    coverage_count = max(len(shuffled_ligands), len(shuffled_receptors))
    for index in range(min(edge_count, coverage_count)):
        pair = (
            int(shuffled_ligands[index % len(shuffled_ligands)]),
            int(shuffled_receptors[index % len(shuffled_receptors)]),
        )
        if pair not in chosen_set:
            chosen.append(pair)
            chosen_set.add(pair)
    remaining = [pair for pair in all_pairs if pair not in chosen_set]
    for position in rng.permutation(len(remaining))[: edge_count - len(chosen)]:
        chosen.append(remaining[int(position)])
    result = pd.DataFrame(chosen, columns=["regulator", "target"])
    return result[["target", "regulator"]]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-cci-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=20260831)
    parser.add_argument("--effect-min", type=float, default=0.3)
    args = parser.parse_args()
    if not 0 <= args.effect_min < 1:
        raise ValueError("effect-min must lie in [0, 1)")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for old_path in sorted(args.old_cci_root.glob("geneNum_*_*.txt")):
        match = PATTERN.fullmatch(old_path.name)
        if not match:
            continue
        gene_num, dataset_id = map(int, match.groups())
        desired_count = len(pd.read_csv(old_path, sep="\t"))
        ligand_count, receptor_count = choose_pool_sizes(gene_num, desired_count)
        rng_seed = stable_seed(args.base_seed, gene_num, dataset_id)
        rng = np.random.default_rng(rng_seed)
        gene_order = rng.permutation(np.arange(1, gene_num + 1))
        ligands = gene_order[:ligand_count]
        receptors = gene_order[ligand_count:ligand_count + receptor_count]
        selected = sample_true_pairs(ligands, receptors, desired_count, rng)
        # Moderate positive effects avoid numerically negligible truth edges.
        selected["effect"] = args.effect_min + (1 - args.effect_min) * rng.beta(2, 2, size=len(selected))
        selected["effect"] = selected["effect"].round(6)
        if selected.duplicated(["target", "regulator"]).any():
            raise AssertionError("redesigned CCI contains duplicate pairs")
        if (selected["target"] == selected["regulator"]).any():
            raise AssertionError("redesigned CCI contains self pairs")
        selected.to_csv(args.output_root / old_path.name, sep="\t", index=False)
        selected.rename(columns={"target": "receptor", "regulator": "ligand"}).to_csv(
            args.output_root / f"lr_pairs_{gene_num}_{dataset_id}.csv", index=False
        )

        candidate = pd.MultiIndex.from_product(
            [ligands.astype(int), receptors.astype(int)], names=["ligand", "receptor"]
        ).to_frame(index=False)
        true_set = set(zip(selected["regulator"], selected["target"]))
        candidate["truth"] = [int(pair in true_set) for pair in zip(candidate["ligand"], candidate["receptor"])]
        effect_lookup = dict(zip(zip(selected["regulator"], selected["target"]), selected["effect"]))
        candidate["effect"] = [effect_lookup.get(pair, 0.0) for pair in zip(candidate["ligand"], candidate["receptor"])]
        candidate.to_csv(
            args.output_root / f"candidate_lr_pairs_{gene_num}_{dataset_id}.tsv", sep="\t", index=False
        )
        roles = pd.DataFrame({
            "gene": np.concatenate([ligands, receptors]).astype(int),
            "role": ["ligand"] * len(ligands) + ["receptor"] * len(receptors),
        })
        roles.to_csv(args.output_root / f"gene_roles_{gene_num}_{dataset_id}.tsv", sep="\t", index=False)
        rows.append({
            "gene_num": gene_num,
            "dataset_id": dataset_id,
            "old_rows": desired_count,
            "new_rows": len(selected),
            "ligand_pool_size": ligand_count,
            "receptor_pool_size": receptor_count,
            "candidate_pair_count": len(candidate),
            "negative_pair_count": int((candidate["truth"] == 0).sum()),
            "effect_min": float(selected["effect"].min()),
            "effect_max": float(selected["effect"].max()),
            "rng_seed": rng_seed,
        })
    audit = pd.DataFrame(rows).sort_values("dataset_id")
    audit.to_csv(args.output_root / "redesign_audit.tsv", sep="\t", index=False)
    manifest = {
        "design": "independent_disjoint_ligand_receptor_pools_with_true_subset_of_cross_product",
        "case_count": len(audit),
        "base_seed": args.base_seed,
        "effect_distribution": f"{args.effect_min} + {1-args.effect_min} * Beta(2, 2)",
        "total_true_edges": int(audit["new_rows"].sum()),
        "total_candidate_pairs": int(audit["candidate_pair_count"].sum()),
        "total_explicit_negative_pairs": int(audit["negative_pair_count"].sum()),
        "row_counts_preserved": bool((audit["old_rows"] == audit["new_rows"]).all()),
        "duplicate_pairs": 0,
        "self_pairs": 0,
        "relationship_to_grn": "independent_gene_role_layer_not_conditioned_on_intracellular_TF_target_edges",
    }
    (args.output_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
