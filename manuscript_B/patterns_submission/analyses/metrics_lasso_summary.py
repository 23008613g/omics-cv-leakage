"""Summary of metrics_lasso.py: accuracy, balanced accuracy and the LASSO pipeline.

Null subset (100 datasets, formal permutations 0-19): each dataset's mean over the 20 permutations, then medians
over datasets with percentile bootstrap 95% CIs (2,000 resamples, seed 20261018). Observed labels (1,086 datasets):
medians of each protocol's value and of the gap (pre-split minus fold-contained). For every method and metric.
Also: the number of AUC values of the formal methods that differ from the formal run (must be 0), and for the LASSO
the features kept by the pre-split fit at each penalty is not recorded (selection is refitted per fold).
Output: results/metrics_lasso_summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extra_analyses import med_ci

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
METHODS = ["top_k", "rfe", "spls", "pls_da", "opls_da", "lasso"]
METRICS = ["auc", "accuracy", "balanced_accuracy"]


def frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        for v in rec["values"]:
            for m in METHODS:
                for arm in ("pre_split", "fold_contained"):
                    rows.append({"dataset_id": rec["dataset_id"], "labels": str(v["labels"]), "method": m, "arm": arm,
                                 **{k: v[m][arm][k] for k in METRICS}})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    out = {}
    obs = df[df.labels == "observed"]
    null = df[df.labels != "observed"].groupby(["dataset_id", "method", "arm"])[METRICS].mean().reset_index()
    for m in METHODS:
        row = {}
        for metric in METRICS:
            o = obs[obs.method == m].pivot(index="dataset_id", columns="arm", values=metric)
            nu = null[null.method == m].pivot(index="dataset_id", columns="arm", values=metric)
            row[metric] = {"observed_pre_split": med_ci(o.pre_split), "observed_fold_contained": med_ci(o.fold_contained),
                           "observed_gap": med_ci(o.pre_split - o.fold_contained),
                           "null_pre_split": med_ci(nu.pre_split), "null_fold_contained": med_ci(nu.fold_contained),
                           "null_gap": med_ci(nu.pre_split - nu.fold_contained)}
        out[m] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="summarize the checkpoint (for testing)")
    args = ap.parse_args()
    if args.partial:
        parts = [json.loads(line) for line in (OUT / "metrics_lasso.partial.jsonl").read_text().splitlines()]
        recs = {}
        for p in parts:
            recs.setdefault(p["dataset_id"], {"dataset_id": p["dataset_id"], "values": []})["values"] += p["values"]
        records, mismatches = list(recs.values()), None
    else:
        data = json.loads((OUT / "metrics_lasso_records.json").read_text())
        records, mismatches = data["records"], data["mismatches_with_formal_auc"]
    df = frame(records)
    summary = {"datasets_observed": int(df[df.labels == "observed"].dataset_id.nunique()),
               "datasets_null": int(df[df.labels != "observed"].dataset_id.nunique()),
               "mismatches_with_formal_auc": mismatches, "methods": summarize(df)}
    name = "metrics_lasso_summary_partial.json" if args.partial else "metrics_lasso_summary.json"
    (OUT / name).write_text(json.dumps(summary, indent=1))
    for m, r in summary["methods"].items():
        print(f"{m:8s} null bacc pre {r['balanced_accuracy']['null_pre_split']['median']:.3f} fc "
              f"{r['balanced_accuracy']['null_fold_contained']['median']:.3f} | null auc pre "
              f"{r['auc']['null_pre_split']['median']:.3f} fc {r['auc']['null_fold_contained']['median']:.3f} | obs gap auc "
              f"{r['auc']['observed_gap']['median']:.3f} bacc {r['balanced_accuracy']['observed_gap']['median']:.3f}")
    print(summary["datasets_observed"], "observed datasets,", summary["datasets_null"], "null datasets; mismatches", mismatches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
