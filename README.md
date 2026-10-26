# TMENet

TMENet jointly estimates cell-type-specific intracellular regulatory networks and spatial cell-cell coupling from spatial expression data and regulatory priors. TMENet-LR subsequently attributes expression-supported coupling to ligand-receptor (LR) pairs. Curated receptor-to-TF routes and receiver-cell co-detection can then connect these attributions to inferred regulatory edges.

This repository contains a portable Python interface, the reference optimization cores, a small synthetic example, tests, and curated research workflows. It does not contain patient-level data, unpublished manuscripts, server credentials, or performance-targeted stress-tuning scripts.

## Install

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
```

The core CPU dependencies are NumPy, SciPy, pandas and scikit-learn. CPU installation does not require a GPU, PyTorch, R, or any benchmark package.

For GPU execution, separately install the CuPy distribution compatible with your NVIDIA driver/CUDA environment following [the official CuPy installation guide](https://docs.cupy.dev/en/stable/install.html). For example, `cupy-cuda12x` is for a compatible CUDA 12 environment. GPU requests fail explicitly if CUDA is unavailable; they do not silently report CPU execution as GPU execution.

## Quick start

A small, generated example is included in `examples/toy/`. It is an installation/API fixture, not one of the manuscript's benchmark results.

```bash
tmenet fit --input examples/toy/inputs.npz --config configs/toy.json --output results/toy --backend cpu
tmenet lr --expression examples/toy/expression.tsv --coupling results/toy/C.npy --metadata examples/toy/metadata.tsv --pairs examples/toy/lr_pairs.tsv --mask examples/toy/candidate_mask.npy --output results/toy_lr.tsv
```

The fit writes `B.npy`, `C.npy`, `X.npy`, `factors.npz`, labels, configuration, objective traces and stopping diagnostics together. Finite max-iteration/nonconverged outputs are retained and labeled; a completed output is not automatically a converged output. Existing nonempty output directories are not overwritten.

The LR command writes sender-receiver-LR summaries and an attribution-conservation audit. Use `--symmetrize` for the manuscript's simulation LR convention; leave it off for directed human-data interpretation.

To regenerate a new toy fixture:

```bash
python examples/make_toy.py --output examples/toy_regenerated
```

## Your own data

Provide expression (genes by spots), metadata, coordinates, a regulatory edge list, and an unlabeled LR candidate list:

```bash
tmenet prepare --expression data/expression.tsv --metadata data/metadata.tsv --coordinates data/coordinates.tsv --prior-edges data/prior_edges.tsv --lr-pairs data/lr_pairs.tsv --output data/inputs.npz
tmenet fit --input data/inputs.npz --config configs/ami.json --output results/my_run --backend cpu
```

See [input and output formats](docs/INPUT_OUTPUT.md). Regulatory priors can come from ATAC peak/motif/target integration or a curated database. The package does not require per-spot paired ATAC measurements. The Gabriel-graph rule selects spatial K adaptively unless an explicit K or spatial mask is supplied.

The supported regularization exponents are `0`, `1/3`, `1/2`, `2/3`, `1`, and `2`:

```bash
tmenet fit --input examples/toy/inputs.npz --config configs/toy.json --p 1/2 --output results/toy_p_half
```

Fractional exponents use the exact scalar proximal map from the six-p experiment, not the older approximate thresholding function.

## Interpretation and reproducibility

- `B` is the regulatory factor; the public API uses **target rows, regulator columns**. Export or transpose explicitly when a downstream tool expects regulator rows. Imported generalized arrays are not silently transposed.
- `C` is a **sender-row, receiver-column coupling matrix**, not an LR-binding probability matrix. LR attribution is a separate post-inference calculation.
- Candidate cross-cell pathways require curated receptor-TF route evidence plus receiver-context co-detection. An IRX6-like intracellular module without an upstream route is not promoted to a complete cross-cell pathway.
- F1 in benchmark scripts is the maximum along the precision-recall curve (`f1_max`), an evaluation summary rather than a deployable label-free classification threshold.

[Methods](docs/METHODS.md) explains the distinction between factor inference, LR attribution and pathway prioritization. [Reproducibility](docs/REPRODUCIBILITY.md) describes AMI/HCC profiles, simulation strata, mixed iteration budgets and optional baseline environments. Research scripts are kept under `reproducibility/`; run them from the repository root with their documented external data and dependencies.

## Tests

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

See `VERIFICATION.md` for the checks actually performed on this release. GPU and third-party baseline checks are not claimed when their external runtimes were unavailable.

## Data, citation and license

Human data are obtained from their original public repositories; see [data access](docs/DATA_AVAILABILITY.md). Re-distribution of those data and third-party databases must follow their own terms.

`CITATION.cff` identifies this software release. No manuscript DOI or publication venue is invented.

**No license is declared for this repository.** The authors have not selected an open-source license. Third-party software and data retain their own licenses; see [third-party resources](THIRD_PARTY.md).

中文说明见 [README_zh-CN.md](README_zh-CN.md)。
