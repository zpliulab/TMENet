# Release verification — 26 October 2026

The portable package was built as a wheel and installed normally in an isolated Windows CPU environment (Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, pandas 3.0.1, scikit-learn 1.9.1, pytest 8.4.2). **All 26 tests passed.** They cover all six p values, reference-core numerical agreement for p=1, joint GRN/CCC output retention, both real-data projection profiles, weighted-projection feasibility, label alignment, adaptive spatial masks, exact fractional proximal maps, LR contribution conservation and reference-evaluator agreement, curated-route filtering, random-prior invariants, permutation/noise mass preservation, and the complete 9,000-record budget ledger.

The installed command-line example completed fitting and LR attribution. Both the random-ATAC refit and ten-repeat coupling-permutation-plus-sigma=2-noise example completed. Twenty command-line smoke checks passed, including help interfaces for optional baseline runners. Python sources were syntax-checked; documentation links, file sizes, private-path patterns and credential-assignment patterns were checked before packaging. These automated scans are not a comprehensive security audit.

A retained max-iteration result produces an intentional runtime warning, not a failed test or a claim of convergence. The bundled example is synthetic and does not reproduce a manuscript performance number.

CUDA/CuPy execution, R-based preprocessing/baselines, optional Dictys/SpaGRN/MDIC3/CellPhoneDB package execution, and full human/simulation reruns were **not executed in this local packaging check**. Their inclusion is not a claim that every external environment has been revalidated. No manuscript or experimental result file was modified by this packaging task.
