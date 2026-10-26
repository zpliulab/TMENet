# Input and output formats

Run commands from the repository root. Tab-separated files have headers. Gene and spot identifiers must be unique, nonempty and consistent across files; gene-symbol aliases are not resolved automatically. Do not include ground-truth labels or effect sizes in inference input files.

## Labeled input route

`tmenet prepare` takes the following tables:

| File | Required structure |
| --- | --- |
| expression.tsv | First column: gene ID; subsequent columns: spot IDs. Values: finite, nonnegative expression/counts. |
| metadata.tsv | `cell`, `cell_type`; exactly one row per expression spot. |
| coordinates.tsv | First column: spot ID; next two columns: x and y. Distinct finite coordinates. |
| prior_edges.tsv | `regulator`, `target`, optionally `cell_type`. Edge existence is encoded as binary prior support. A missing cell-type column applies the same scaffold to all types. |
| lr_pairs.tsv | `ligand`, `receptor` only; an unlabeled candidate universe, not the positive truth list. |

The preparation command groups spots by sorted cell-type label while preserving within-type expression-column order. For m genes, n spots and t types, it constructs:

- `A`: (tm, n), cell-type expression blocks on the diagonal, structural zeros elsewhere.
- `B_prior`, `B_mask`: (tm, tm), with cell-type blocks. The public edge-list preparation uses **target rows and regulator columns**.
- `C_prior`: (n, n), sender rows and receiver columns; binary LR co-detection support restricted to the spatial mask.
- `M_knn`: (n, n), zero-diagonal binary directed mask; kth-distance ties are retained.
- `D`, `coordinates`, and string arrays `genes`, `cells`, `cell_types`, `block_types`.

The `.npz` bundle contains no pickled objects. A neighboring `.spatial.json` records adaptive K and realized neighborhood sizes. The generalized matrices are dense: memory grows approximately quadratically in tm and n. Filter features and use appropriate compute resources before processing large datasets.

## Already assembled matrices

Use `fit_generalized` in Python or assemble an NPZ with `A`, `B_prior`, `C_prior`, `B_mask` and `M_knn` (or coordinates). Set `normalize` in the JSON configuration to `false` if the generalized expression has already undergone the reference log1p/min-max transformation. Do not normalize twice.

**Legacy orientation is not automatically detected.** The original simulation preprocessing and GRN benchmark helpers use regulator-by-target edge matrices. The public preparation function uses the target-by-regulator convention of the manuscript equation. Import and transpose prior/output blocks explicitly when converting between these formats; do not mix an old generalized prior with new labels or silently transpose fitted results. The archived six-p inference/export convention is documented separately in [reproducibility](REPRODUCIBILITY.md).

## Fit outputs

Each run saves `B.npy`, `X.npy`, `C.npy`, a compressed `factors.npz`, `objective.json`, and `diagnostics.json`. The CLI additionally saves `labels.json` and `run_config.json`. Labels record the actual fit order, not the original input order. A nonempty existing output directory is rejected. `converged`, `iterations`, and `stop_reason` must accompany performance summaries; max-iteration stopping does not discard a finite result.

## LR attribution

`tmenet lr` takes expression, metadata, LR candidates, a fitted coupling matrix, and preferably the exact frozen inference/evaluation mask. A `.npy` coupling file must have adjacent `labels.json`; a TSV/CSV coupling matrix has row and column labels. A supplied `.npy` mask must already be in the coupling order recorded in `labels.json`, not the unsorted expression order.

Use `--symmetrize` only for the simulation evaluation convention. Human-data analyses retain sender-to-receiver direction. If no mask is supplied, positive coupling entries define the candidate edges; this can omit zero-valued candidates and is **not** the formal simulation evaluation universe.

The score table has `sender_type`, `receiver_type`, `ligand`, `receptor`, `attributed_sum`, `attributed_mean`, `attribution_share_percent`, `support_edges`, and candidate/support mass fields. `attributed_mean` divides by **all** spatial candidates in a type pair, not just positive expression-supported edges. The `.audit.json` records missing candidate genes, unsupported coupling mass, and per-edge conservation error.

## Candidate cross-cell pathways

`tmenet pathways` additionally requires:

- A curated route table: `receptor`, `tf`, `route`, `source_database`. A route may be multistep; store its path and database provenance explicitly.
- A receiver GRN edge table: `cell_type`, `regulator`, `target`, `weight`.

Only positive LR contributions, positive GRN edges, database-supported receptor-to-TF routes, and receptor/TF co-detection in the receiver context are joined. The default co-detection threshold is max(20 spots, ceil(1% of receiver spots)); toy examples with fewer spots need an explicit smaller `--min-codetected`. Prioritization is descriptive, not a causal-effect estimate. A TF-to-target module without a supported upstream route remains an intracellular module.
