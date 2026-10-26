# Implementation notes

## Joint inference

The package wraps the audited ADMM-based nonnegative joint factorization cores. It estimates B (cell-type regulatory structure), X (latent expression representation), and C (spatial coupling) together. The manuscript and supplementary derivation define the objective and constraints; the executable reference implementations are `_core_cpu.py` and `_core_real.py`. This release reorganizes code and adds input validation, configuration and output handling; it does not refit the manuscript's biological conclusions during packaging.

The simulation profile uses the frozen six-p optimizer. For p=1/3, 1/2 and 2/3, `_lp_prox.py` supplies the exact scalar proximal operator used in that experiment. p=0, 1 and 2 retain their corresponding closed forms. Finite iterates are saved whether stopping is by convergence or iteration budget.

The real-data profile uses the AMI/HCC core. HCC additionally enables the audited corrected weighted-ball projection in `_hcc_projection.py`; the AMI profile does not silently adopt this HCC-specific switch. There is no new X-mask algorithm in this interface. Profiles are starting configurations for the recorded data transformations, not universal optimal hyperparameters for unrelated data.

## Adaptive spatial support

For distinct two-dimensional coordinates, a closed-ball Gabriel graph is constructed. K is round(mean Gabriel degree), clipped to [3,12], with effective K at most n-1. Directed KNN retains all kth-distance ties; realized degrees can therefore exceed K. An explicit K or binary mask can override this rule. Tiny/collinear datasets use an exact all-pairs geometric check; large degenerate geometries require explicit support rather than a silent fallback.

## Coupling-conserving LR attribution

Expression is converted to within-gene percentile ranks over positive values; zero expression stays zero. For candidate LR pair (l,r) and spatial sender-receiver edge (i,j), q(l,r,i,j) is the sender ligand rank times receiver receptor rank. Its contribution is:

`attribution(l,r,i,j) = C[i,j] * q(l,r,i,j) / sum_over_candidate_LR(q)`.

When the denominator is zero, all LR attributions on that edge are zero and its coupling is reported as unsupported. Otherwise the contributions sum to C[i,j]. This explicitly retains the inferred coupling magnitude, unlike a test-only conversion of coupling to LR significance. Type-pair scores average over the same frozen spatial candidate edges, including zeros. The simulation evaluator uses symmetrized coupling; human interpretation preserves direction.

The candidate LR list is not a positive label list. Injected simulator effects and realized single-cell CCI truth are available only to evaluation. A ligand/receptor need not be a TF or be regulated in the input GRN.

## Receptor-to-TF evidence

For biological interpretation, receptor-to-TF routes must come from a curated signaling resource such as OmniPath/Reactome and retain route/database evidence. Receiver receptor/TF co-detection provides contextual support, not proof of signal transmission. These routes are joined to LR attribution and inferred TF-target edges. No causal, self-locking, or experimentally validated feedback interpretation is inferred solely from this join.

## Evaluation and ablations

GRN evaluation uses the declared directed TF-to-nonself-target universe. CCC comparison is at sender type × receiver type × LR resolution, not a mixture of group and spot-pair outputs. Successful no-prediction cases are explicitly recorded before unpredicted candidates receive zero; technical failures are not treated as successful zero predictions. `f1_max` is the maximum on the precision-recall curve and uses evaluation labels.

The random-ATAC control rewires off-diagonal prior edges while preserving directed row/column degrees, row weight multisets, diagonals and block structure. Identical input blocks share one randomization. Seeds depend on case identity, not performance. This is followed by joint refitting; expression, spatial support and all other settings stay fixed.

The CCC post-hoc control permutes inferred symmetric coupling within reciprocal/one-way spatial-candidate strata, then applies fixed sigma=2 multiplicative lognormal noise and matches total candidate coupling mass. LR attribution is rerun with expression, candidate LR identities and spatial support fixed. This is a **permutation-plus-noise stress ablation**, not pure permutation, a retrained model, or a complete sigma sensitivity curve. The helper supports individual sigma values for diagnostics; their presence does not establish that a full sensitivity experiment was reported.
