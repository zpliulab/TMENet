# Third-party software and resources

No open-source license has been selected for TMENet in this release. This does not change any third-party license or grant redistribution rights over external databases.

The portable Python interface depends on NumPy, SciPy, pandas and scikit-learn. GPU execution optionally uses CuPy. Their distributions are installed separately; no vendored wheels or compiled binaries are included.

The optional research scripts call the following upstream tools, which must be installed in their own compatible environments:

| Workflow | External tools |
| --- | --- |
| Simulations | scMultiSim (recorded study version 1.1.10), ape, jsonlite |
| Classical GRN baselines | scikit-learn; R minet and GENIE3 |
| Specialized GRN baselines | Dictys; SpaGRN and its dependencies |
| CCC baselines | CellChat + jsonlite; CellPhoneDB; MDIC3 |
| Custom-prior NicheNet | nichenetr, Seurat, dplyr, jsonlite |
| HCC prior pruning | R infotheo |
| Human upstream ATAC processing | Cicero, genome annotations and motif resources (not distributed here) |

The NicheNet protocol records upstream commit `66f90d5eeafef280b2b2f339b3fd70ffec1781dd`. Other historical environments were not captured as a single complete lockfile; the core release requirements are install constraints, not a claim to recreate every original baseline environment exactly. Research runs should save `sessionInfo()` or package version manifests.

Official project references include [scMultiSim](https://github.com/JSB-UCLA/scMultiSim), [NicheNet](https://github.com/saeyslab/nichenetr), [CellChat](https://github.com/jinworks/CellChat), [CellPhoneDB](https://github.com/ventolab/CellphoneDB), and [Dictys](https://github.com/pinellolab/dictys). Consult the applicable upstream license and cite the relevant method papers when using their outputs.

TMENet's LR attribution is its own coupling-conserving calculation. MDIC3-LR's official significance-based calculation remains separately identified in the research evaluator. No third-party LR database or executable is included in this repository.
