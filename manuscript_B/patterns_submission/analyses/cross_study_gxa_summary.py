"""Summary of cross_study_gxa.py with the statistics of the MicrobiomeHD analysis (reserve/summarize_reserve.py:
medians, percentile bootstrap 95% CIs resampling contrasts, shares, Spearman correlations, permuted-label control).
The two breast cancer platforms form one contrast for resampling. Output: results/cross_study_gxa_summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reserve"))
import summarize_reserve as SR                            # noqa: E402

OUT = HERE / "results"


def frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        base = {"contrast": r["cluster"], "train": r["train"], "test": r["test"], "n_train": r["n_train"],
                "n_test": r["n_test"], "genera_shared": r["genera_shared"]}
        for m, v in r["methods"].items():
            rows.append({**base, "method": m, "labels": "observed",
                         **{k: v["observed"][k] for k in ("pre_split", "fold_contained", "external")}})
            rows += [{**base, "method": m, "labels": f"perm{p['labels']}",
                      **{k: p[k] for k in ("pre_split", "fold_contained", "external")}} for p in v["permuted"]]
    return pd.DataFrame(rows)


def main() -> int:
    records = json.loads((OUT / "cross_study_gxa_records.json").read_text())
    summary = SR.cross_cohort(frame(records))
    summary["platform_contrasts"] = sorted({r["contrast"] for r in records})
    summary["overlap"] = json.loads((OUT / "cross_study_gxa_overlap.json").read_text())
    (OUT / "cross_study_gxa_summary.json").write_text(json.dumps(summary, indent=1))
    for m, r in summary["methods"].items():
        print(f"{m:8s} CV pre {r['pre_split']['median']:.3f} fc {r['fold_contained']['median']:.3f} ext "
              f"{r['external']['median']:.3f} | error pre {r['error_pre_split']['median']:.3f} fc "
              f"{r['error_fold_contained']['median']:.3f} | closer {r['fold_contained_closer']['share']:.2f} | null pre "
              f"{r['null']['pre_split']['median']:.3f} fc {r['null']['fold_contained']['median']:.3f} ext "
              f"{r['null']['external']['median']:.3f}")
    print(summary["pairs"], "pairs,", summary["contrasts"], "contrasts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
