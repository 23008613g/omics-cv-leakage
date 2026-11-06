"""Summarise ``results/validation_records.json`` into the numbers the manuscript cites."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
STEPS = ("top_k", "rfe", "spls", "pls_da", "opls_da")


def main() -> None:
    records = json.loads((HERE / "results" / "validation_records.json").read_text())
    sample = json.loads((HERE / "results" / "sample.json").read_text())
    failed = [r["dataset_id"] for r in records if "error" in r]
    ok = [r for r in records if "error" not in r]
    out = {"seed": sample["seed"], "n_exchangeable": sample["n_exchangeable"], "n_datasets": len(records),
           "n_failed": len(failed), "failed": failed, "n_null_datasets": len(sample["null_ids"]),
           "null_indices": [0, 1, 2, 3, 4], "median_seconds_per_dataset": float(np.median([r["seconds"] for r in ok])),
           "methods": {}}
    for m in STEPS:
        obs, null, own, formal = [], [], [], []
        for r in ok:
            x = r["methods"][m]
            obs += [x["exact_pre_split"] - x["formal_pre_split"], x["exact_fold_contained"] - x["formal_fold_contained"]]
            null += [d for z in x.get("null", []) for d in (z["exact_pre_split"] - z["formal_pre_split"],
                                                            z["exact_fold_contained"] - z["formal_fold_contained"])]
            own.append(x["own_pre_split"] - x["own_fold_contained"])
            formal.append(x["formal_pre_split"] - x["formal_fold_contained"])
        obs, null, own, formal = map(np.asarray, (obs, null, own, formal))
        secs = [r["methods"][m]["seconds"] for r in ok]
        out["methods"][m] = {
            "median_seconds_per_dataset": float(np.median(secs)),
            "observed_values_compared": int(obs.size), "observed_max_abs_diff": float(np.abs(obs).max()),
            "observed_identical": int(np.count_nonzero(obs == 0)),
            "null_values_compared": int(null.size), "null_max_abs_diff": float(np.abs(null).max()),
            "null_identical": int(np.count_nonzero(null == 0)),
            "own_folds_spearman_gap": float(spearmanr(own, formal)[0]),
            "median_gap_formal": float(np.median(formal)), "median_gap_own_folds": float(np.median(own)),
            "median_abs_gap_difference": float(np.median(np.abs(own - formal))),
        }
    (HERE / "results" / "validation_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
