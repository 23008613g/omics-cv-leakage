"""Summary of the outer-fold sensitivity analysis (cv_folds_sensitivity.py): 10 against 5 outer folds.

For each method, on the 100-dataset null subset: the pre-split and fold-contained nulls (mean over formal
permutations 0-19), the gap between them under the null, and the real-label gap and fold-contained AUC, under 5
folds (formal values: Data S1 and the null check, Table S3) and under 10 folds, with the median paired difference
(10 minus 5). Dataset-level percentile bootstrap 95% CIs (2,000 resamples, seed 20261018), as in the other
exploratory tables. Output: results/cv_folds_summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from extra_analyses import METHODS, SD1, med_ci

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def nullcheck_nulls() -> pd.DataFrame:
    """Five-fold nulls of the 100-dataset subset: the arm check's means over formal permutations 0-19."""
    nc = pd.read_csv(OUT / "nullcheck_per_dataset.csv")
    assert set(nc.n_perm) == {20}
    return (nc[nc.arm.isin(["pre_split", "fold_contained"])]
            .pivot_table(index=["dataset_id", "key"], columns="arm", values="null_mean").reset_index()
            .rename(columns={"key": "method", "pre_split": "null_pre_5", "fold_contained": "null_fc_5"}))


def checkpoint_nulls(ids) -> pd.DataFrame:
    """Five-fold values of any dataset from the formal checkpoints: observed AUCs and the means over permutations
    0-19 (full precision; Data S1 rounds the last digit of some values)."""
    import sys
    sys.path.insert(0, str(HERE.parent / "validation"))
    import validate_against_formal as V
    rows = []
    for i in ids:
        vals = {}
        for block in V.checkpoint(i)["permutation_blocks"]:
            res = block["result"]
            for m in METHODS:
                for k, a, b in zip(res["permutation_indices"], res["methods"][m]["pre_split_oof_auc"],
                                   res["methods"][m]["fold_contained_oof_auc"], strict=True):
                    vals[(m, int(k))] = (a, b)
        observed = V.checkpoint(i)["observed"]["methods"]
        for m in METHODS:
            perm = [vals[(m, k)] for k in range(20)]
            rows.append({"dataset_id": i, "method": m, "null_pre_5": float(np.mean([a for a, _ in perm])),
                         "null_fc_5": float(np.mean([b for _, b in perm])),
                         "pre_5": observed[m]["observed"]["pre_split_oof_auc"],
                         "fc_5": observed[m]["observed"]["fold_contained_oof_auc"]})
    return pd.DataFrame(rows)


def paired_table(records: list[dict], nulls: pd.DataFrame | None = None) -> pd.DataFrame:
    ok = [r for r in records if "error" not in r]
    ten = pd.DataFrame([{"dataset_id": r["dataset_id"], "method": m, "outer_folds": r["outer_folds"], "n": r["n"],
                         "p": r["p"], "pre_10": v["pre_split"], "fc_10": v["fold_contained"],
                         "null_pre_10": v["null_pre_split"], "null_fc_10": v["null_fold_contained"]}
                        for r in ok for m, v in r["methods"].items()])
    if nulls is None:
        nulls = nullcheck_nulls()
    df = ten.merge(nulls, on=["dataset_id", "method"], how="left", validate="one_to_one")
    if "pre_5" not in df:                                   # 100-dataset subset: observed values from Data S1
        s1 = pd.read_csv(SD1)
        obs = pd.concat([pd.DataFrame({"dataset_id": s1.dataset_id, "method": m, "pre_5": s1[f"{m}_pre_split_auc"],
                                       "fc_5": s1[f"{m}_fold_contained_auc"]}) for m in METHODS])
        df = df.merge(obs, on=["dataset_id", "method"], how="left", validate="one_to_one")
    assert df[["null_pre_5", "null_fc_5", "pre_5", "fc_5"]].notna().all().all(), "missing 5-fold values"
    return df


def summarize(df: pd.DataFrame, n_records: int, n_errors: int, null_ids=None) -> dict:
    """Paired summaries. Null quantities use the datasets in ``null_ids`` (all if None); observed-label
    quantities use every dataset."""
    out = {"n_records": n_records, "n_errors": n_errors, "methods": {}}
    for m in METHODS:
        d = df[df.method == m]
        dn = d if null_ids is None else d[d.dataset_id.isin(null_ids)]
        row = {"n": int(len(d)), "n_null": int(len(dn)), "reduced_folds": int((d.outer_folds < 10).sum())}
        for name, (five, ten), x in (("null_pre", ("null_pre_5", "null_pre_10"), dn),
                                     ("null_fc", ("null_fc_5", "null_fc_10"), dn), ("fc", ("fc_5", "fc_10"), d)):
            row[name] = {"five": med_ci(x[five]), "ten": med_ci(x[ten]), "diff": med_ci(x[ten] - x[five]),
                         "rho": float(spearmanr(x[five], x[ten]).statistic)}
        for name, (pre, fc), x in (("null_gap", ("null_pre", "null_fc"), dn), ("gap", ("pre", "fc"), d)):
            g5, g10 = x[f"{pre}_5"] - x[f"{fc}_5"], x[f"{pre}_10"] - x[f"{fc}_10"]
            row[name] = {"five": med_ci(g5), "ten": med_ci(g10), "diff": med_ci(g10 - g5),
                         "rho": float(spearmanr(g5, g10).statistic)}
        out["methods"][m] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(OUT / "cv_folds_records.json"))
    ap.add_argument("--out", default=str(OUT / "cv_folds_summary.json"))
    ap.add_argument("--all", action="store_true", help="all 1,086 datasets (cv_folds_all.py); five-fold nulls from "
                    "the formal checkpoints, null quantities over the 1,030 datasets with exchangeable labels")
    args = ap.parse_args()
    if args.all:
        args.records, args.out = str(OUT / "cv_folds_all_records.json"), str(OUT / "cv_folds_all_summary.json")
    records = json.loads(Path(args.records).read_text())
    errors = [r for r in records if "error" in r]
    null_ids = None
    if args.all:
        nulls = checkpoint_nulls([r["dataset_id"] for r in records if "error" not in r])
        check = nullcheck_nulls().merge(nulls, on=["dataset_id", "method"], suffixes=("", "_ck"))
        diff = float(np.max(np.abs(check[["null_pre_5", "null_fc_5"]].to_numpy()
                                   - check[["null_pre_5_ck", "null_fc_5_ck"]].to_numpy())))
        assert len(check) == 500 and diff < 1e-12, f"checkpoint and arm-check nulls differ by {diff}"
        s1 = pd.read_csv(SD1)
        null_ids = set(s1[s1.classification == "eligible_primary_formal"].dataset_id)
        df = paired_table(records, nulls)
    else:
        df = paired_table(records)
    summary = summarize(df, len(records), len(errors), null_ids)
    same = df[df.outer_folds < 5]                           # fold count capped below five: the formal plan itself
    summary["same_plan"] = {"datasets": int(same.dataset_id.nunique()),
                            "identical": bool(all((same[f"{q}_10"] == same[f"{q}_5"]).all()
                                                  for q in ("pre", "fc", "null_pre", "null_fc")))}
    reduced = df[df.outer_folds < 10].outer_folds
    summary["reduced_fold_range"] = [int(reduced.min()), int(reduced.max())] if reduced.size else None
    df.to_csv(Path(args.out).with_name(Path(args.out).stem.replace("summary", "per_dataset") + ".csv"), index=False)
    Path(args.out).write_text(json.dumps(summary, indent=1))
    for m, r in summary["methods"].items():
        print(f"{m:8s} n={r['n']} null pre 5 {r['null_pre']['five']['median']:.3f} 10 {r['null_pre']['ten']['median']:.3f}"
              f" diff {r['null_pre']['diff']['median']:+.3f} | null fc diff {r['null_fc']['diff']['median']:+.3f}"
              f" | gap 5 {r['gap']['five']['median']:.3f} 10 {r['gap']['ten']['median']:.3f}"
              f" diff {r['gap']['diff']['median']:+.3f}")
    print("errors:", [e["dataset_id"] for e in errors])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
