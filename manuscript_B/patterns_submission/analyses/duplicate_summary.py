"""Summary of the duplicate scan and its sensitivity analysis (duplicate_scan.py, duplicate_sensitivity.py).

Counts from the scan; per dataset, the rerun with same-label duplicates kept together against its control on the
same group-aware path (medians over the five methods); the datasets in which keeping duplicates together lowered a
fold-contained AUC by more than 0.05; the size of fold-assignment noise (control against formal); and the largest
change of a corpus-level median when the formal values of the rerun datasets are replaced by the duplicate-grouped
ones. Output: results/duplicate_summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
DATA_S1 = HERE.parent / "output" / "Patterns_Data_S1_per_dataset.csv"     # Data S1 of the article (build_patterns.py)
METHODS = ["top_k", "rfe", "spls", "pls_da", "opls_da"]
DROP = 0.05


def main() -> int:
    scan = json.loads((OUT / "duplicate_scan.json").read_text())
    recs = scan["records"]
    sens = json.loads((OUT / "duplicate_sensitivity.json").read_text())
    out = {"datasets": len(recs), "near": scan["near"],
           "exact": {"datasets": sum(r["exact_pairs"] > 0 for r in recs), "pairs": sum(r["exact_pairs"] for r in recs)},
           "near_any": sum(r["near_pairs"] > 0 for r in recs),
           "unhandled": sum(r["unhandled_pairs"] > 0 for r in recs),
           "cross_label": {"datasets": sum(r["unhandled_cross_label"] > 0 for r in recs),
                           "pairs": sum(r["unhandled_cross_label"] for r in recs)},
           "exact_cross_label": sorted(r["dataset_id"] for r in recs
                                       if any(not p["same_label"] and p["exact"] for p in r["pairs"]))}
    rows, noise = [], []
    for d, r in sens.items():
        ok = r["cv"] is not None and r["cv"]["status"] == "ok"
        row = {"dataset_id": d, "n": r["scan"]["n"], "exact_pairs": r["scan"]["exact_pairs"],
               "same_label_pairs": r["same_label_pairs"], "median_nearest_r": r["scan"]["median_nearest_r"],
               "status": r["cv"]["status"] if r["cv"] else "missing",
               "reason": None if ok or r["cv"] is None else r["cv"].get("reason")}
        if ok:
            dup, ctl = r["cv"]["methods"], r["cv_control"]["methods"]
            for arm in ("fold_contained", "pre_split"):
                row[f"{arm}_control"] = float(np.median([ctl[m][arm] for m in METHODS]))
                row[f"{arm}_duplicates"] = float(np.median([dup[m][arm] for m in METHODS]))
            row["gap_control"] = float(np.median([ctl[m]["pre_split"] - ctl[m]["fold_contained"] for m in METHODS]))
            row["gap_duplicates"] = float(np.median([dup[m]["pre_split"] - dup[m]["fold_contained"] for m in METHODS]))
            row["largest_fc_drop"] = float(max(ctl[m]["fold_contained"] - dup[m]["fold_contained"] for m in METHODS))
            noise += [abs(ctl[m]["fold_contained"] - r["formal"][m]["fold_contained"]) for m in METHODS]
        rows.append(row)
    df = pd.DataFrame(rows)
    ev = df[df.status == "ok"]
    hit = ev[ev.largest_fc_drop > DROP]
    out["rerun"] = {"datasets": len(df), "evaluable": len(ev), "not_evaluable": df[df.status != "ok"][
        ["dataset_id", "reason"]].to_dict(orient="records"),
        "noise_q90": float(np.quantile(noise, 0.9)),
        "affected": {"threshold": DROP, "datasets": len(hit),
                     "fold_contained": [float(hit.fold_contained_control.median()), float(hit.fold_contained_duplicates.median())],
                     "pre_split": [float(hit.pre_split_control.median()), float(hit.pre_split_duplicates.median())],
                     "gap": [float(hit.gap_control.median()), float(hit.gap_duplicates.median())],
                     "most_samples_near_identical": int((hit.median_nearest_r >= 0.95).sum())},
        "per_dataset": df.sort_values("largest_fc_drop", ascending=False, na_position="last").to_dict(orient="records")}
    # corpus-level medians with the rerun values in place of the formal ones
    s1 = pd.read_csv(DATA_S1)
    s1 = s1[s1.top_k_pre_split_auc.notna()].set_index("dataset_id")
    shift = 0.0
    for m in METHODS:
        ho = s1[s1[f"{m}_holdout_heldout_auc"].notna()]
        series = {"gap": s1[f"{m}_pre_split_auc"] - s1[f"{m}_fold_contained_auc"], "fc": s1[f"{m}_fold_contained_auc"],
                  "hold_pre": ho[f"{m}_holdout_pre_split_cv_auc"] - ho[f"{m}_holdout_heldout_auc"],
                  "hold_fc": ho[f"{m}_holdout_fold_contained_cv_auc"] - ho[f"{m}_holdout_heldout_auc"]}
        for key, x in series.items():
            y = x.copy()
            for d, r in sens.items():
                if key in ("gap", "fc") and r["cv"] and r["cv"]["status"] == "ok":
                    v = r["cv"]["methods"][m]
                    y[d] = v["pre_split"] - v["fold_contained"] if key == "gap" else v["fold_contained"]
                if key.startswith("hold") and r["holdout"] and r["holdout"]["status"] == "ok" and d in y.index:
                    h = r["holdout"]["result"]["methods"][m]
                    arm = "pre_split_oof_auc" if key == "hold_pre" else "fold_contained_oof_auc"
                    y[d] = h[arm] - h["heldout_auc"]
            shift = max(shift, abs(float(y.median() - x.median())))
    out["corpus_shift"] = shift
    out["holdout_reruns"] = {"datasets": sum(r["holdout"] is not None for r in sens.values()),
                             "ok": sum(r["holdout"] is not None and r["holdout"]["status"] == "ok" for r in sens.values())}
    (OUT / "duplicate_summary.json").write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k not in ("rerun",)}, {k: v for k, v in out["rerun"].items() if k != "per_dataset"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
