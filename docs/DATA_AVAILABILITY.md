# Data and external resources

This code-only release includes a small generated toy dataset. It does not redistribute human expression matrices, patient identifiers/clinical records, large simulation outputs, or third-party database dumps. External data remain subject to their original access conditions.

Sources recorded in the current manuscript:

- Human AMI atlas: [CZ CELLxGENE collection](https://cellxgene.cziscience.com/collections/8191c283-0816-424b-9b61-c3e1d6258a77), [Zenodo record](https://zenodo.org/record/6578047), and [Human Cell Atlas project](https://data.humancellatlas.org/explore/projects/e9f36305-d857-44a3-93f0-df4e6007dc97).
- Human HCC spatial sections HCC-2L and HCC-1L: HCCDB, at the manuscript-recorded portal `http://lifeome.net:809/#/home`.
- TCGA-LIHC expression and clinical data: [GDC Data Portal](https://portal.gdc.cancer.gov/). The study's recorded retrieval date is 13 October 2026.
- Human LR candidates: the study integrated CellChat, CellPhoneDB and NicheNet lists. The historical integration did not record a single originating database/version for each row. Do not present that list as a newly curated or uniquely versioned resource. Obtain upstream resources under their own terms, record versions and normalize identifiers before rebuilding a list.
- Regulatory/signaling resources: RegNetwork for HCC regulatory scaffolds; curated OmniPath/Reactome receptor-to-TF routes for post-hoc pathway evidence. Database routes are not replaced by coexpression alone.

For AMI, the spatial input sections were Control/P7, Acute/P9 (ischaemic zone), and Fibrosis/P14 (fibrotic zone), with 468 shared genes. The matched ATAC workflow combined peak co-accessibility, genomic/TSS annotation and TF motif matches into binary, cell-type-specific TF-target priors. This repository accepts the resulting edge lists; it does **not** bundle a complete peak-calling/Cicero/motif reference-genome workflow or the unpublished laboratory manuscript. Those upstream steps require the genome build, annotations and motif resources used with the original data.

For HCC, the main section used 417 genes and five cell types. The corrected PCA-CMI pruning implementation is included under `reproducibility/preprocessing/`. Human feature lists and processed matrices are external inputs, not silently reconstructed from toy data.

The simulation scripts expect the study's 100 weighted input GRNs and designed CCI tables. These source networks and the 1,500 generated paired datasets are not fabricated or replaced in this release. Each case is named `geneNum_N_id_I/cellTypeT/seedS`, with T=2,3,4 and S=12,22,32,42,52. The included 9,000-row iteration ledger describes six p profiles over these 1,500 cases; it is run metadata, not expression data or benchmark scores.
