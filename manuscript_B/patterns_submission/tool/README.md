# presplit-audit

How much does a classifier's cross-validated AUC depend on where its supervised feature step was fitted?

`presplit-audit` scores one dataset under two protocols on the same folds, with the same tuning choice and classifier:

- **pre-split**: label-free preprocessing and the supervised step (feature selection or projection) are fitted once on all samples; only the classifier is refitted in each fold;
- **fold-contained**: preprocessing, the supervised step and the classifier are all refitted on each training partition.

It reports both pooled out-of-fold AUCs and their gap, and, on request, a full-pipeline permutation test for each protocol: every permutation rebuilds the split plan and reruns both complete pipelines, so each protocol is compared with a null that carries its own optimism.

Report the fold-contained AUC as the performance estimate. A pre-split AUC is not a performance estimate; its permutation P value tests association only.

## Install

```bash
pip install .            # from this folder; requires Python >= 3.10, numpy, scipy, scikit-learn, pandas, joblib
pip install ".[test]" && pytest
```

## Use

```python
from presplit_audit import audit

result = audit(X, y, "top_q", groups=participant_ids, n_permutations=199, random_state=0, n_jobs=4)
print(result.summary())
```

Command line, for a CSV with one row per sample:

```bash
presplit-audit data.csv --label phenotype --group subject --drop sample_id --step pls_da \
    --permutations 199 --jobs 4 --json result.json
```

Built-in supervised steps, with the grids of the benchmark in the accompanying article: `top_q` (two-sample F
statistic, q in {5, 10, 20, 50}), `rfe` (recursive feature elimination with a linear SVM, C = 0.1, to 50, 20, 10 and
5 features), `ht_spls` (hard-threshold sparse PLS; 1 or 2 components keeping 5-50 features), `pls_da` (1, 2, 3 or 5
latent components), `opls_da` (one predictive and 0-3 orthogonal components) and `vip` (PLS-DA VIP > 1). Any step
can be wrapped:

```python
from presplit_audit import CustomStep, audit
from sklearn.feature_selection import mutual_info_classif
import numpy as np

def mi_top(X, y, k):
    keep = np.argsort(-mutual_info_classif(X, y, random_state=0))[:k]
    return lambda Z: Z[:, np.sort(keep)]

audit(X, y, CustomStep("mutual_information", mi_top, [{"k": 10}, {"k": 50}]))
```

The default preprocessor (`StandardPreprocessor`) removes features observed in fewer than half of the samples,
imputes medians, drops constant features, keeps the 5,000 highest-variance features and z-scores, all estimated
from the rows it is fitted on. Put data on an appropriate scale first (for example, log-transform intensities),
or pass your own object with `fit(X)` and `transform(X)` methods as `preprocessor=`.

## Defaults and design

- Outer folds: stratified 5-fold, repeated 3 times; inner folds: stratified 3-fold within each training partition.
  With `groups`, stratified group k-fold keeps each participant in one fold in both loops.
- The candidate setting with the highest mean inner AUC (ties to the earlier candidate) is used by both protocols.
- Classifier: L2-penalized logistic regression (C = 1, liblinear). Any `seed -> estimator` factory with
  `fit` and `predict_proba` can be passed as `classifier=`.
- Each sample's out-of-fold score is averaged over repeats before one AUC is computed.
- Permutations exchange samples, or participants when labels are constant within participant, or labels within
  participants otherwise. P values are (b + 1)/(B + 1).
- A custom split plan can be supplied through `splitter=` (a function of the labels and groups returning
  `OuterFold` objects), and a seed schedule through `evaluate_protocols(..., seeds=)`, which maps each fitting
  context to a seed; by default every fit uses one seed (only `rfe` is affected, as liblinear's dual solver
  shuffles samples).

## How high can noise push a pre-split AUC?

Without any data, `presplit-risk` returns a design-only estimate of the AUC that a pre-split pipeline typically
reaches on pure noise for a given number of samples and features, with the range that held 80% of the reference
datasets:

```bash
presplit-risk --samples 56 --features 1875
```

```python
from presplit_audit import expected_presplit_null
expected_presplit_null(56, 1875, "pls_da")   # {'auc': ..., 'range80': (..., ...), 'supported': True}
```

The estimate comes from an exploratory model fitted to 1,030 public omics datasets (accompanying article,
Tables S17 and S18). It is a rough expectation for the built-in pipelines, transfers imperfectly across platforms
(especially for PLS-DA and OPLS-DA) and is not a test: use `audit` with permutations to test a dataset.

## Validation

Given the split plans, fold preprocessing, seed schedule and permuted labels of the benchmark in the accompanying
article, the tool reproduced all 2,000 AUCs checked (all five methods, both protocols, observed labels in 100
datasets and five permutations in 20 of them) exactly. See `validation/` in the archive.

## Scope

Binary classification and the AUC only. The tool measures how an estimate depends on the evaluation protocol;
it does not validate a biomarker or replace an independent test set.

## Licence

BSD 3-Clause (see `LICENSE`).


## Authors and licences

Author and copyright holder: Cheng Zhang.
Software source code is licensed under BSD 3-Clause (`LICENSE`). The author-generated
risk lookup data in `src/presplit_audit/data/risk_grid.json` are licensed under
Creative Commons Attribution 4.0 International (CC BY 4.0; `DATA_LICENSE`).
Third-party data retain their original licences and terms.
