"""Does the human disease observational subset result change with the re-review's inclusion set?

Mirrors ``strict_human_complement`` in the manuscript builder (same tables, seed and
bootstrap), once with the original inclusion set and once with the 258 queued
datasets decided by the re-review instead of the original review.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TABLES = HERE.parents[1] / "Formal_Public_Deposit_v3" / "tables"
METHODS = ["top_k", "rfe", "spls", "pls_da", "opls_da"]


def compare(meta: pd.DataFrame, inc: set[str]) -> dict:
    rg = np.random.default_rng(20260930)
    out = {}
    for m in METHODS:
        d = meta[meta.method == m]
        a = pd.to_numeric(d[d.dataset_id.isin(inc)].gap).to_numpy()
        b = pd.to_numeric(d[~d.dataset_id.isin(inc)].gap).to_numpy()
        reps = [np.median(rg.choice(a, a.size)) - np.median(rg.choice(b, b.size)) for _ in range(2000)]
        out[m] = {"n_in": int(a.size), "n_out": int(b.size), "median_in": float(np.median(a)),
                  "median_out": float(np.median(b)), "diff": float(np.median(a) - np.median(b)),
                  "diff_ci": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]}
    return out


def main() -> None:
    meta = pd.read_csv(TABLES / "meta_input.csv", low_memory=False)
    h = pd.read_csv(TABLES / "human_cohort_adjudication_v2_manual.csv", low_memory=False)
    queued = set(h.loc[h.reviewer_1_decision.notna(), "id"])
    original = set(h.loc[h.strict_observational_status.eq("include") | h.manual_final_status.eq("include"), "id"])
    rr = pd.read_csv(HERE / "rereview_decisions.csv")
    automatic = original - queued
    rereview = automatic | set(rr.loc[rr.decision.eq("include"), "id"])
    out = {"n_automatic_include": len(automatic), "n_original_include": len(original),
           "n_rereview_include": len(rereview), "original": compare(meta, original),
           "rereview": compare(meta, rereview)}
    (HERE / "subset_sensitivity.json").write_text(json.dumps(out, indent=1))
    for m in METHODS:
        o, r = out["original"][m], out["rereview"][m]
        print(f"{m:8s} original {o['median_in']:.3f} vs {o['median_out']:.3f} (diff {o['diff']:+.3f}) | "
              f"re-review {r['median_in']:.3f} vs {r['median_out']:.3f} (diff {r['diff']:+.3f}, "
              f"CI {r['diff_ci'][0]:+.3f} to {r['diff_ci'][1]:+.3f}) n_in {o['n_in']}->{r['n_in']}")


if __name__ == "__main__":
    main()
