# Research workflows

There are two entry routes. Use the public `tmenet` commands for labeled new data and `reproducibility/simulation/run_six_p.py` for the study's **legacy generalized matrices and export convention**. Do not interchange prior orientation or output formats between these routes. The research workflow is primarily intended for Linux with optional R/CUDA environments; CPU package installation and toy tests are portable to Windows.

## Scope and recorded profiles

The release contains the frozen six-p simulation core, the current real-data core, corrected HCC PCA-CMI pruning and weighted-ball projection, LR evaluation helpers, and formal baseline runners. No performance-targeted Dictys stress settings, private server launchers, biological data, old notebooks with saved outputs, or manuscript PDFs are distributed.

`configs/ami.json` records the shared AMI real-data parameter profile with adaptive neighborhoods and 1,000 maximum iterations. `configs/hcc.json` records the HCC profile with the corrected global weighted-ball projection, lambda_X=0.01 and 2,000 iterations. The HCC main fit used seed 42; seeds 7 and 2026 were additional stability runs. Relative prior-radius settings (main 0.90; comparison 0.80) were converted to absolute radii for the corresponding processed matrices. **Do not apply those absolute radii to unrelated matrices and assume the same relative constraint.** These are reference profiles, not automated parameter-selection results for a new dataset.

For a new human dataset, construct its own labeled expression and edge-list inputs, then run `tmenet prepare`, `tmenet fit`, `tmenet lr`, and (with external database routes) `tmenet pathways`. AMI's upstream Cicero/motif workflow and human feature/route tables are not reconstructed automatically; see [data availability](DATA_AVAILABILITY.md).

## Simulation generation and priors

The scripts preserve the case hierarchy and coarse type/cell counts. Starting from the original 100 weighted GRNs and original CCI edge-count templates:

1. `simulation/design_cci.py --help` describes CCI role assignment, candidate-universe design and output paths. Ligand/receptor roles are independent of intracellular GRN edges. The simulator receives true effects; inference receives identities of true and decoy candidates without labels.
2. `simulation/generate_scmultisim.R` generates the paired RNA/ATAC cases using scMultiSim and saves realized CCI truth. Call `Rscript reproducibility/simulation/generate_scmultisim.R GRN_ROOT CCI_ROOT OUTPUT_ROOT [SHARD_COUNT SHARD_INDEX]`; preserve seeds and case structure when rebuilding data.
3. `simulation/prepare_scmultisim.py --root ... --grn-root ...` constructs ATAC-derived priors and directed GRN truth exports. Region-to-TF, mean regional accessibility and region-to-gene mappings determine the prior; the GRN truth is a separate evaluation export, not a prior-construction input.
4. `simulation/export_scmultisim_lr_truth.R` exports realized LR ground truth. `prepare_simulation_ccc.py` is the legacy fixed-K candidate/prior preparation utility; it does not select adaptive K for inference. The joint runner recomputes geometry-derived K and projects the legacy C prior onto that mask.

The R generator's usage is defined in its final command-line block. Run all scripts from the repository root so relative defaults resolve consistently. Source weighted GRNs/CCI templates are external study inputs; the toy fixture is not substituted for them.

## Six-p joint runs and iteration budgets

`configs/simulation_strata.json` contains the fixed coefficients for small (10–40 genes), medium (41–70), and large (71–100) strata. The 9,000-row `simulation/iteration_budgets.csv` records the actual final per-case budgets, iterations and stopping states across all six p values. The study initially allowed extended runs and later capped remaining jobs at 1,000 iterations; this was not a single uniform-budget experiment. Do not discard finite nonconverged records or describe all cases as converged.

```bash
python reproducibility/simulation/run_six_p.py --sim-root data/simulation/simdata_v3 --truth-root data/simulation/grn_networks --output results/six_p --p-label p1
```

Omit `--p-label` for all six exponents, or repeat it to select several. `--case-key geneNum_10_id_1/cellType2/seed12` selects one case. The runner uses the recorded ledger budgets by default; `--max-iter` creates an explicitly different new-run budget. Small networks run on CPU and medium/large networks require CuPy/CUDA, matching the recorded hybrid scheduling policy. Every successful joint fit saves block-level GRN scores, raw and symmetric coupling, and diagnostics in one case directory. It does not overwrite an existing case metric file.

`legacy_joint.py` retains the original normalization, initialization, scoring/export bodies and legacy regulator-by-target block convention. `legacy_io.py` contains only three unchanged extracted I/O/mask helpers. `optimizer_adapter.py` lazily loads the same core on CPU/CuPy and applies the recorded exact fractional proximal operator. No truth is supplied to the optimizer; evaluation occurs after fitting.

## GRN baselines

`benchmarks/run_python_classical.py` supports its declared correlation/regression/ensemble methods. For example:

```bash
python reproducibility/benchmarks/run_python_classical.py --method pcc --sim-root data/simulation/simdata_v3 --truth-root data/simulation/grn_networks --output-root results/baselines/pcc --limit 1
```

`run_python_specialized.py` runs **formal** Dictys or SpaGRN in separately installed environments. No reverse-tuned stress configurations are included. `run_r_classical.R` calls minet/GENIE3. All use `grn_benchmark_common.py` for directed, nonself TF-target evaluation. Finite-score filtering in SpaGRN is explicit; a technical case failure is not relabeled a successful zero prediction.

## CCC baselines and common LR evaluation

The primary unit is sender fine type × receiver fine type × LR candidate. Native group scores are not expanded into artificial independent spot-pair measurements.

```bash
Rscript reproducibility/benchmarks/run_cellchat_group_native.R data/simulation/simdata_v3 data/simulation/ccc_networks results/cellchat geneNum_10_id_1/cellType2/seed12
python reproducibility/benchmarks/run_cpdb_group_native.py --sim-root data/simulation/simdata_v3 --db-root data/cpdb_custom --output-root results/cpdb --case-key geneNum_10_id_1/cellType2/seed12
python reproducibility/benchmarks/benchmark_lr_contribution_v2.py --sim-root data/simulation/simdata_v3 --lr-root data/simulation/ccc_networks --tmenet-root results/six_p/p1 --cellchat-root results/cellchat --cpdb-root results/cpdb --output-root results/ccc --case-key geneNum_10_id_1/cellType2/seed12
```

`build_cpdb_simulated_databases.py` builds custom CellPhoneDB databases from unlabeled candidates and an external template database. MDIC3-LR's official sorted-expression Pearson/Bonferroni calculation is included in the LR evaluator. The optional MDIC3 coupling runner in `run_python_ccc_baselines.py` and its distance/expression controls are **legacy cell-coupling diagnostics**, not the manuscript's primary LR comparison. An MDIC3 C matrix passed through TMENet's attribution is labeled separately from official MDIC3-LR.

The LR evaluator reads adaptive K from TMENet case metrics unless `--k` explicitly overrides it. The primary simulation score is `tmenet_symmetric_lr_conserved_attribution`; raw/transpose-directed scores are diagnostics. Evaluation files retain truth effects and realized truth density, distinct from the unlabeled inference list.

## Custom-prior NicheNet

```bash
python reproducibility/nichenet_custom/launch.py --sim-root data/simulation/simdata_v3 --frozen-eval-root results/ccc --output results/nichenet_custom --workers 1
python reproducibility/nichenet_custom/evaluate_custom.py --root results/nichenet_custom --workers 4
```

The formal launcher requires 1,500 completed cases; `run_custom_batch.R` can also be called directly with a smaller case list for an installation smoke test. On Windows use one R worker; multi-process R execution is intended for Linux. Install the recorded nichenetr revision separately with Seurat/dplyr/jsonlite. The launcher does not install or silently change R packages.

`protocol.json` records the custom LR prior, ATAC TF-target prior and **absolute partial-Spearman association proxy** from receptors to active TFs. The proxy globally ranks each gene and centers ranks within fine cell types; it is not a curated signaling network or causal route. It is distinct from the database-constrained route interpretation used for human TMENet pathways. Native PPR propagation and native one-condition LR prioritization are used. Frozen truth/baseline files are read only during subsequent paired evaluation, never during custom-prior construction or inference.

## Ablations

Random ATAC prior, preserving other inputs and optionally refitting:

```bash
python reproducibility/ablations/run_ablation.py random-atac --input examples/toy/inputs.npz --genes-per-block 8 --case toy --config configs/toy.json --output results/toy_random_atac
```

For the paper's original generalized inputs, use the corresponding legacy prior orientation and identical fit settings; the randomizer preserves both in/out degrees regardless of the declared orientation. Inspect `mixing_target_reached` and changed-support counts; constrained tiny graphs may not permit meaningful rewiring.

Post-hoc coupling permutation plus sigma=2 noise, with ten prespecified deterministic repeats:

```bash
python reproducibility/ablations/run_ablation.py coupling --fit results/toy --expression examples/toy/expression.tsv --metadata examples/toy/metadata.tsv --pairs examples/toy/lr_pairs.tsv --mask examples/toy/candidate_mask.npy --case toy --output results/toy_coupling_control
```

This control keeps the fitted GRN fixed and is not a joint-model refit. Optional `--truth` performs evaluation **after** all predictions are saved. Differences are `100 × (full minus control)` in percentage points, not relative percentages. Negative/small differences and all repeats are retained. No code chooses sigma or seeds to reach a desired performance threshold. Sigma=2 is the specific configured perturbation; do not present it as a data-independent biological null or a full sensitivity curve.

## Limits of this release check

Packaging checks do not rerun the 1,500 datasets, all baselines, or human case studies. Exact numerical equality across CPU/GPU and dependency versions is not guaranteed. Record versions, inputs, configurations, masks, seeds, stopping states and hashes for new runs. See `VERIFICATION.md` for tests actually performed and `SOURCE_PROVENANCE.json` for source identifiers and hashes.
