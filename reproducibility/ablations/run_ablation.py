"""Portable truth-blind controls; evaluation is optional and strictly post-hoc."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tmenet.cli import main as tmenet_main, read_frame, read_table
from tmenet.lr import attribute_lr
from tmenet.metrics import binary_metrics
from randomize_prior import randomize_prior
from permutation import permute_coupling
from coupling import add_n


KEYS = ["sender_type", "receiver_type", "ligand", "receptor"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate(truth, scores):
    if not set(KEYS + ["truth_present"]).issubset(truth):
        raise ValueError("truth requires sender_type/receiver_type/ligand/receptor/truth_present")
    if truth.duplicated(KEYS).any() or scores.duplicated(KEYS).any():
        raise ValueError("duplicate LR evaluation keys")
    table = truth[KEYS + ["truth_present"]].merge(
        scores[KEYS + ["attributed_mean"]], on=KEYS, how="left", validate="one_to_one")
    if table.attributed_mean.isna().any():
        raise ValueError("truth universe not covered by inferred candidate scores; do not silently impute")
    return binary_metrics(table.truth_present.to_numpy(), table.attributed_mean.to_numpy())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="control", required=True)
    a = sub.add_parser("random-atac", help="randomize B*; optionally refit the joint model")
    a.add_argument("--input", required=True); a.add_argument("--case", required=True)
    a.add_argument("--genes-per-block", type=int, required=True)
    a.add_argument("--output", required=True); a.add_argument("--config")
    a.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    c = sub.add_parser("coupling", help="post-hoc symmetric permutation plus fixed sigma=2 noise")
    c.add_argument("--fit", required=True); c.add_argument("--expression", required=True)
    c.add_argument("--metadata", required=True); c.add_argument("--pairs", required=True)
    c.add_argument("--mask", required=True); c.add_argument("--case", required=True)
    c.add_argument("--output", required=True); c.add_argument("--truth")
    args = parser.parse_args(argv)
    dest = Path(args.output)
    if dest.exists(): raise FileExistsError(f"refusing to overwrite {dest}")
    dest.mkdir(parents=True)
    if args.control == "random-atac":
        if args.genes_per_block < 1: raise ValueError("genes-per-block must be positive")
        with np.load(args.input, allow_pickle=False) as handle:
            data = {key: handle[key] for key in handle.files}
        prior, audit = randomize_prior(data["B_prior"], args.genes_per_block, args.case)
        data["B_prior"] = prior
        with (dest / "randomized_inputs.npz").open("wb") as handle:
            np.savez_compressed(handle, **data)
        write_json(dest / "randomization.json", audit)
        if args.config:
            tmenet_main(["fit", "--input", str(dest / "randomized_inputs.npz"),
                         "--config", args.config, "--output", str(dest / "fit"),
                         "--backend", args.backend])
        return
    fitted = Path(args.fit)
    cells = json.loads((fitted / "labels.json").read_text(encoding="utf-8"))["cells"]
    expression = read_frame(args.expression)
    expression.columns = expression.columns.astype(str)
    if set(cells) != set(expression.columns): raise ValueError("fit/expression cell labels differ")
    expression = expression.loc[:, cells]
    metadata, pairs = read_table(args.metadata), read_table(args.pairs)
    mask = np.load(args.mask, allow_pickle=False)
    raw = np.load(fitted / "C.npy", allow_pickle=False)
    coupling = (raw + raw.T) / 2
    full, audit = attribute_lr(expression, coupling, metadata, pairs, candidate_mask=mask)
    full.to_csv(dest / "full_lr.tsv", sep="\t", index=False)
    write_json(dest / "full_audit.json", audit)
    paths = [("full", dest / "full_lr.tsv")]
    for repeat in range(10):
        permuted, perm_audit = permute_coupling(coupling, mask, args.case, repeat)
        noisy, noise_audit = add_n(permuted, mask, args.case, repeat, 2.0)
        scores, attr_audit = attribute_lr(expression, noisy, metadata, pairs, candidate_mask=mask)
        name = f"repeat_{repeat:02d}"
        score_path = dest / f"{name}_lr.tsv"
        scores.to_csv(score_path, sep="\t", index=False)
        write_json(dest / f"{name}_audit.json", {
            "case": args.case, "repeat": repeat, "sigma": 2.0,
            "control": "posthoc_symmetric_permutation_plus_lognormal_noise",
            "truth_accessed_for_randomization": False,
            "permutation": perm_audit, "noise": noise_audit, "attribution": attr_audit})
        paths.append((name, score_path))
    # Ground truth is first read only after all randomized predictions are saved.
    if args.truth:
        truth = read_table(args.truth)
        for key in KEYS: truth[key] = truth[key].astype(str)
        rows = []
        for name, path in paths:
            scores = pd.read_csv(path, sep="\t", dtype={key: str for key in KEYS})
            rows.append({"condition": name, **evaluate(truth, scores)})
        frame = pd.DataFrame(rows)
        full_values = frame.iloc[0]
        for metric in ["auroc", "average_precision", "f1_max"]:
            frame[f"full_minus_control_{metric}_pp"] = 100 * (full_values[metric] - frame[metric])
        frame.to_csv(dest / "paired_metrics.tsv", sep="\t", index=False)


if __name__ == "__main__": main()
