#!/usr/bin/env python3
"""Build one topology-faithful CellPhoneDB database per simulated LR network."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

def canonical_gene(value: object) -> str:
    number = float(str(value).strip())
    return str(int(number)) if number.is_integer() else str(value).strip()

def read_pairs(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t" if path.suffix.lower() in {".txt", ".tsv"} else ",")
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    frame = frame.rename(columns={"regulator": "ligand", "target": "receptor"})
    if not {"ligand", "receptor"}.issubset(frame.columns):
        raise ValueError(f"invalid LR columns: {path}")
    result = frame[["ligand", "receptor"]].copy()
    result["ligand"] = result["ligand"].map(canonical_gene)
    result["receptor"] = result["receptor"].map(canonical_gene)
    return result.drop_duplicates().reset_index(drop=True)

def empty_like(template_root: Path, relative_path: str) -> pd.DataFrame:
    return pd.DataFrame(columns=pd.read_csv(template_root / relative_path, nrows=0).columns)

def build_database(
    gene_num: int,
    dataset_id: int,
    pairs: pd.DataFrame,
    template_root: Path,
    output_root: Path,
) -> dict[str, object]:
    from cellphonedb.utils import db_utils
    output_dir = output_root / f"geneNum_{gene_num}_id_{dataset_id}"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and list(output_dir.glob("cellphonedb_*.zip")):
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    genes = [str(index) for index in range(1, gene_num + 1)]
    ligands = set(pairs["ligand"])
    receptors = set(pairs["receptor"])
    uniprot = {gene: f"SIMU{int(gene):05d}" for gene in genes}

    gene_input = empty_like(template_root, "gene_input.csv")
    for gene in genes:
        gene_input.loc[len(gene_input)] = {
            "gene_name": f"g{gene}",
            "uniprot": uniprot[gene],
            "hgnc_symbol": f"g{gene}",
            "ensembl": "",
        }

    protein_input = empty_like(template_root, "protein_input.csv")
    for gene in genes:
        is_ligand = gene in ligands
        is_receptor = gene in receptors
        row = {column: "" for column in protein_input.columns}
        row.update(
            {
                "uniprot": uniprot[gene],
                "protein_name": f"SIM_GENE_{gene}",
                "transmembrane": is_receptor,
                "peripheral": False,
                "secreted": is_ligand,
                "secreted_highlight": is_ligand,
                "receptor": is_receptor,
                "integrin": False,
                "biosynthetic_pathway": False,
                "other": False,
                "version": "simulated_topology_v1",
            }
        )
        protein_input.loc[len(protein_input)] = row

    interaction_input = empty_like(template_root, "interaction_input.csv")
    for ligand, receptor in pairs.itertuples(index=False, name=None):
        row = {column: "" for column in interaction_input.columns}
        row.update(
            {
                "partner_a": uniprot[ligand],
                "partner_b": uniprot[receptor],
                "annotation_strategy": "curated",
                "source": "simulated_LR_topology",
                "is_ppi": True,
                "curator": "TMENet_benchmark",
                "version": "simulated_topology_v1",
                "interactors": f"g{ligand}-g{receptor}",
                "classification": "simulated_ligand_receptor",
                "directionality": "Ligand-Receptor",
            }
        )
        interaction_input.loc[len(interaction_input)] = row

    # CellPhoneDB 5's database builder cannot process a truly empty complex
    # table (it calls np.apply_along_axis on zero rows).  Add one inert orphan
    # complex that is never referenced by an interaction.
    complex_input = empty_like(template_root, "complex_input.csv")
    dummy_complex = {column: "" for column in complex_input.columns}
    dummy_complex.update(
        {
            "complex_name": "SIM_DUMMY_COMPLEX_NOT_USED",
            "uniprot_1": uniprot[genes[0]],
            "uniprot_2": uniprot[genes[0]],
            "uniprot_3": uniprot[genes[0]],
            "uniprot_4": uniprot[genes[0]],
            "uniprot_5": uniprot[genes[0]],
            "transmembrane": False,
            "peripheral": False,
            "secreted": False,
            "secreted_highlight": False,
            "receptor": False,
            "integrin": False,
            "other": True,
            "pdb_structure": False,
            "curator": "TMENet_benchmark",
            "version": "simulated_topology_v1",
        }
    )
    complex_input.loc[len(complex_input)] = dummy_complex
    tf_input = empty_like(template_root, "sources/transcription_factor_input.csv")
    gene_input.to_csv(output_dir / "gene_input.csv", index=False)
    protein_input.to_csv(output_dir / "protein_input.csv", index=False)
    interaction_input.to_csv(output_dir / "interaction_input.csv", index=False)
    complex_input.to_csv(output_dir / "complex_input.csv", index=False)
    tf_input.to_csv(output_dir / "transcription_factor_input.csv", index=False)
    db_utils.create_db(str(output_dir))

    zip_path = sorted(output_dir.glob("cellphonedb_*.zip"))[-1]
    extracted = db_utils.extract_dataframes_from_db(str(zip_path))
    interaction_count = len(extracted["interaction_table"])
    if interaction_count != len(pairs):
        raise ValueError(
            f"CellPhoneDB retained {interaction_count}/{len(pairs)} interactions for geneNum={gene_num}, id={dataset_id}"
        )
    manifest = {
        "gene_num": gene_num,
        "dataset_id": dataset_id,
        "gene_count": gene_num,
        "lr_pair_count": len(pairs),
        "database_zip": str(zip_path),
        "topology_policy": "exact_simulated_gene_and_LR_identity_no_human_remapping",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-root", type=Path)
    parser.add_argument("--lr-root", type=Path, default=Path("data/simulation/ccc_networks"))
    parser.add_argument(
        "--template-root", type=Path, default=Path("data/simulation/cpdb_benchmark/database_v5")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    tasks: list[tuple[tuple[int, int], Path]] = []
    if args.sim_root is not None:
        for gene_dir in sorted(args.sim_root.glob("geneNum_*_id_*")):
            match = re.fullmatch(r"geneNum_([0-9]+)_id_([0-9]+)", gene_dir.name)
            if not match:
                continue
            paths = sorted(gene_dir.glob("cellType*/seed*/LR_candidate_universe.tsv"))
            if not paths:
                raise FileNotFoundError(f"no case LR candidate universe under {gene_dir}")
            reference = read_pairs(paths[0])
            reference_set = set(map(tuple, reference.to_numpy()))
            for path in paths[1:]:
                if set(map(tuple, read_pairs(path).to_numpy())) != reference_set:
                    raise ValueError(f"LR candidate universe differs within dataset: {gene_dir}")
            tasks.append(((int(match.group(1)), int(match.group(2))), paths[0]))
    else:
        candidates = sorted(args.lr_root.glob("geneNum_*_*.txt"))
        if not candidates:
            candidates = sorted(args.lr_root.glob("lr_pairs_*_*.csv"))
        raw_tasks = []
        for path in candidates:
            match = re.search(r"(?:geneNum|lr_pairs)_([0-9]+)_([0-9]+)", path.stem)
            if match:
                raw_tasks.append((int(match.group(1)), int(match.group(2)), path))
        tasks = sorted({(gene_num, dataset_id): path for gene_num, dataset_id, path in raw_tasks}.items())
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard-count/shard-index")
    tasks = [
        task for index, task in enumerate(tasks)
        if index % args.shard_count == args.shard_index
    ]
    for position, ((gene_num, dataset_id), path) in enumerate(tasks, start=1):
        print(f"[{position}/{len(tasks)}] geneNum={gene_num} id={dataset_id}", flush=True)
        build_database(gene_num, dataset_id, read_pairs(path), args.template_root, args.output_root)

if __name__ == "__main__":
    main()
