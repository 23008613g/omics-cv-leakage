# presplit-audit

Paired cross-validation audits for supervised feature selection and projection,
with code and derived records from a benchmark of 1,086 public omics datasets.
The software compares pre-split and fold-contained evaluation and tests each
protocol against its own full-pipeline permutation null.

Author: **Cheng Zhang**, School of Chinese Medicine, The University of Hong Kong.
Contact: zttzc@hku.hk.

[Version v0.2.0](https://github.com/23008613g/presplit-audit/releases/tag/v0.2.0) · [Complete derived records](https://github.com/23008613g/presplit-audit/releases/download/v0.2.0/Formal_Public_Deposit_v3.zip) ·
[Citation](CITATION.cff) · [Verification results](VALIDATION.json)

## Install and run

Python 3.14.3 and the versions in `requirements-reproduction.txt` were used for
the recorded results and the release checks. From the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-reproduction.txt
python -m pip install --no-deps -e manuscript_B/patterns_submission/tool
python -m pytest manuscript_B/patterns_submission/tool/tests -q
python manuscript_B/patterns_submission/tool/examples/example_synthetic.py
python reproduce_summaries.py
```

The example generates simulated noise and weak-signal datasets.
`reproduce_summaries.py` rebuilds nine sets of numerical summaries from the
included records and compares them with the stored values at a tolerance of
1e-10. It works in a temporary copy, preserving the archived records.

## Contents

- `manuscript_B/patterns_submission/tool/`: installable software, tests, a synthetic example and risk lookup data.
- `mtbls_screen/`: formal evaluator and source-repository loaders, with the 18 formal code hashes.
- `manuscript_B/Formal_Public_Deposit_v3/tables/`: selected derived benchmark tables; the feature-stability CSV is gzip-compressed.
- `manuscript_B/patterns_submission/{analyses,validation,reserve,rereview}/`: analysis scripts and derived records.
- `manuscript_B/patterns_submission/figures/source_data/`: numerical source data for the figures.
- `manuscript_B/nm_brief_communication/control_arm/`: code and records for additional benchmark analyses.

The directory names follow the research code's import and data-path structure.
The package includes dataset-level and sample-level derived records, including
source-repository sample identifiers used for duplicate grouping.

## Complete benchmark records

The [Release archive](https://github.com/23008613g/presplit-audit/releases/download/v0.2.0/Formal_Public_Deposit_v3.zip) contains 12 canonical CSV tables and 1,102
sanitized checkpoints, including fold predictions, permutation summaries and
feature-selection records. Its size and SHA-256 are in `RELEASE_ASSETS.json`;
the Release also provides `SHA256SUMS.txt` and `ASSET_MANIFEST.json`.

Extract the archive into a separate directory, then run:

```sh
unzip /path/to/Formal_Public_Deposit_v3.zip -d ../benchmark_records
export PRESPLIT_FORMAL_DEPOSIT="$(cd ../benchmark_records/Formal_Public_Deposit_v3 && pwd)"
python scripts/validate_formal_deposit.py --root "$PRESPLIT_FORMAL_DEPOSIT"
python reproduce_summaries.py --with-full-checkpoints
```

The additional summary uses the existing checkpoints to compare five and ten
outer folds across all analyzed datasets. A new model-training run requires
the source matrices and repository catalogue caches under their original terms.
These third-party matrices are obtained from the source repositories. The
optional Parquet aggregation path requires `pyarrow`; release verification uses
the CSV path. The default checks run offline; `requests` is used by the loaders.

## Research provenance

AI tools assisted code development, the initial repository-metadata review,
the blinded re-review, and literature-audit scoring. Cheng Zhang checked the
code, result numbers, references and literature-audit scores and reviewed the
258 inclusion decisions. The two review sessions do not constitute independent
human expert validation. `PROVENANCE.json` records this scope.

## Licence and citation

Software: BSD 3-Clause (`LICENSE`). Author-generated derived data and results:
CC BY 4.0 (`DATA_LICENSE`). Third-party source metadata retain their original
terms. Cite this release using `CITATION.cff`.
