"""Merge the blinded re-review and only then compare it with the original review.

The re-review decisions (``decisions_batch*.jsonl`` plus ``second_pass_overrides.jsonl``)
were written before this script first read the original decision columns. This is
agreement between two reviews of the same metadata.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
QUEUE = HERE.parents[1] / "Formal_Public_Deposit_v3" / "tables" / "human_cohort_adjudication_v2_manual.csv"
ORIGINAL = {"decision": "manual_final_status", "human": "human_participant_manual",
            "disease": "manual_disease_like", "intervention": "manual_intervention_exclusion"}


def kappa(a: pd.Series, b: pd.Series) -> float:
    cats = sorted(set(a) | set(b))
    table = pd.crosstab(a, b).reindex(index=cats, columns=cats, fill_value=0).to_numpy(float)
    n = table.sum()
    po = np.trace(table) / n
    pe = float((table.sum(0) * table.sum(1)).sum()) / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def norm(v) -> str:
    s = str(v).strip().lower()
    return {"true": "yes", "false": "no", "1": "yes", "0": "no", "1.0": "yes", "0.0": "no",
            "nan": "missing", "none": "missing"}.get(s, s)


def main() -> None:
    rows = {}
    for f in sorted(HERE.glob("decisions_batch*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            assert r["id"] not in rows, f"duplicate {r['id']}"
            rows[r["id"]] = {**r, "pass": "first"}
    for line in (HERE / "second_pass_overrides.jsonl").read_text().splitlines():
        r = json.loads(line)
        assert r["id"] in rows
        rows[r["id"]] = {**r, "pass": "second (override)"}
    mine = pd.DataFrame(rows.values()).sort_values("id")
    assert len(mine) == 258
    mine.to_csv(HERE / "rereview_decisions.csv", index=False)

    orig = pd.read_csv(QUEUE, usecols=["id", *ORIGINAL.values()])
    m = mine.merge(orig, on="id", how="left", validate="one_to_one")
    out = {"n": len(m), "note": "agreement between two reviews of the same metadata", "criteria": {}}
    print({k: m[v].map(norm).value_counts().to_dict() for k, v in ORIGINAL.items()})
    for key, col in ORIGINAL.items():
        a, b = m[key].map(norm), m[col].map(norm)
        both = (a != "missing") & (b != "missing")
        out["criteria"][key] = {"n_compared": int(both.sum()), "agreement": float((a[both] == b[both]).mean()),
                                "kappa": kappa(a[both], b[both]),
                                "crosstab": pd.crosstab(a[both], b[both]).to_dict()}
    d = m.assign(rer=m.decision.map(norm), ori=m[ORIGINAL["decision"]].map(norm))
    out["rereview_include"] = int((d.rer == "include").sum())
    out["original_include"] = int((d.ori == "include").sum())
    out["disagreements"] = d.loc[d.rer != d.ori, ["id", "rer", "ori", "note"]].to_dict("records")
    (HERE / "agreement_summary.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "disagreements"}, indent=1, default=str))
    print(f"{len(out['disagreements'])} decision disagreements")


if __name__ == "__main__":
    main()
