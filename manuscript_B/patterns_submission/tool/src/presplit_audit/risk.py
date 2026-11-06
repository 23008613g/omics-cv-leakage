"""Design-only estimate of how high a pre-split pipeline pushes the AUC on pure noise.

The estimate comes from an exploratory model fitted to 1,030 public omics datasets in the accompanying article
(fractional logit with smooth terms in log10 n and log10 p, one model per method; Table S17). It predicts the
typical pre-split AUC under permuted labels from the number of samples and the number of features counted before
preprocessing, for the article's pipelines (default preprocessing, four-candidate grids, L2-logistic classifier).
Within that corpus the 10-fold cross-validated mean absolute error was 0.04-0.07 AUC and 80% of datasets fell in
the returned range; transfer across platforms was weaker, especially for PLS-DA and OPLS-DA. Treat the estimate
as a rough expectation, not as a test of a dataset: use ``audit`` with permutations for that.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from functools import lru_cache
from importlib import resources

STEP_TO_MODEL = {"top_q": "top_k", "rfe": "rfe", "ht_spls": "spls", "pls_da": "pls_da", "opls_da": "opls_da"}


@lru_cache(maxsize=1)
def _grid() -> dict:
    return json.loads(resources.files("presplit_audit").joinpath("data/risk_grid.json").read_text())


def _locate(axis: list[float], value: float) -> tuple[int, float] | None:
    if not axis[0] <= value <= axis[-1]:
        return None
    i = min(max(next(k for k in range(1, len(axis)) if axis[k] >= value) - 1, 0), len(axis) - 2)
    return i, (value - axis[i]) / (axis[i + 1] - axis[i])


def expected_presplit_null(n_samples: int, n_features: int, step: str = "pls_da") -> dict:
    """Typical pre-split AUC under permuted labels for a design, with the empirical 80% range.

    Returns ``{"step", "auc", "range80", "supported"}``. ``supported`` is False when the design lies outside the
    region covered by the 1,030 datasets (then ``auc`` is NaN).
    """
    if step not in STEP_TO_MODEL:
        raise ValueError(f"unknown step {step!r}; choose from {sorted(STEP_TO_MODEL)}")
    if n_samples < 1 or n_features < 1:
        raise ValueError("n_samples and n_features must be positive")
    g = _grid()
    model = g["methods"][STEP_TO_MODEL[step]]
    ln, lp = _locate(g["log10_n"], math.log10(n_samples)), _locate(g["log10_p"], math.log10(n_features))
    nan = {"step": step, "auc": float("nan"), "range80": (float("nan"), float("nan")), "supported": False}
    if ln is None or lp is None:
        warnings.warn("design outside the range of the reference datasets (n 30-1,200, p 20-180,000)", stacklevel=2)
        return nan
    (i, a), (j, b) = ln, lp
    corners = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
    if not all(g["supported"][r][c] for r, c in corners):
        warnings.warn("too few reference datasets with a similar design; no estimate returned", stacklevel=2)
        return nan
    v = model["auc"]
    auc = ((1 - a) * (1 - b) * v[i][j] + a * (1 - b) * v[i + 1][j] + (1 - a) * b * v[i][j + 1]
           + a * b * v[i + 1][j + 1])
    lo, hi = model["error_q10_q90"]
    return {"step": step, "auc": float(auc), "range80": (max(0.0, auc + lo), min(1.0, auc + hi)), "supported": True}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="presplit-risk", description=(
        "Typical AUC that a pre-split pipeline reaches on pure noise, estimated from the number of samples and "
        "features (exploratory model from 1,030 public omics datasets)."))
    p.add_argument("--samples", type=int, required=True, help="number of samples (n)")
    p.add_argument("--features", type=int, required=True, help="number of features before preprocessing (p)")
    p.add_argument("--step", choices=sorted(STEP_TO_MODEL), help="one method (default: all five)")
    args = p.parse_args(argv)
    steps = [args.step] if args.step else list(STEP_TO_MODEL)
    print(f"Design: n = {args.samples} samples, p = {args.features:,} features")
    for s in steps:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = expected_presplit_null(args.samples, args.features, s)
        if r["supported"]:
            print(f"  {s:8s} typical pre-split AUC on noise {r['auc']:.2f} (80% of reference datasets "
                  f"{r['range80'][0]:.2f}-{r['range80'][1]:.2f})")
        else:
            print(f"  {s:8s} no estimate: {caught[0].message if caught else 'outside the reference designs'}")
    print("A rough expectation, not a test; transfer across platforms is limited. Use presplit-audit with "
          "permutations to test a dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
