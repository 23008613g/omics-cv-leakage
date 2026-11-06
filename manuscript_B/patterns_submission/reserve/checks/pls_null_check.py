"""Check: is the fold-contained PLS-DA null of the cross-cohort analysis centred on 0.5?

The 10-permutation means of two training studies were 0.543 (crc_xiang) and 0.530 (hiv_dinh). Here 100
permutations (indices 0-99; 0-9 are those of the main run) of the same training data and genera are evaluated
with PLS-DA and top-q, fold-contained and pre-split, to separate noise from bias. Output:
results/checks_pls_null.json.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C                                         # noqa: E402
import cross_cohort as X                                   # noqa: E402

CASES = (("CRC", "CRC", "crc_xiang", "crc_zhao"), ("HIV", "HIV", "hiv_dinh", "hiv_lozupone"))
N = 100


def one(job):
    (contrast, case, a, b), i = job
    cfg = C.formal_config()
    ga, ya, groups_a, det_a = X.cohort(C.load_study(a), case)
    _, _, _, det_b = X.cohort(C.load_study(b), case)
    Xa = ga[sorted(det_a & det_b)].to_numpy()
    unit = f"xcohort|{contrast}|{a}"
    yy = C.permuted(ya, groups_a, unit, i, cfg)
    folds = C.binary_plan(yy, groups_a, unit, cfg)
    out = {}
    for m in ("top_k", "pls_da"):
        r = C.evaluate_generic(Xa, yy, folds, C.make_step("binary", m, cfg), task="binary",
                               preprocessor=C.preprocessor_for(cfg, "microbiome_counts"),
                               seeds=C.V.formal_seeds(cfg, unit, m))
        out[m] = (r["pre_split"], r["fold_contained"])
    return a, i, out


if __name__ == "__main__":
    jobs = [(case, i) for case in CASES for i in range(N)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        res = list(pool.map(one, jobs, chunksize=10))
    summary = {}
    for case in CASES:
        a = case[2]
        rows = [r[2] for r in res if r[0] == a]
        summary[a] = {m: {"fold_contained_mean": float(np.mean([x[m][1] for x in rows])),
                          "fold_contained_se": float(np.std([x[m][1] for x in rows], ddof=1) / np.sqrt(len(rows))),
                          "fold_contained_mean_first10": float(np.mean([x[m][1] for x in rows[:10]])),
                          "pre_split_mean": float(np.mean([x[m][0] for x in rows]))} for m in ("top_k", "pls_da")}
    C.write_json(C.RESULTS / "checks_pls_null.json", summary)
    print(json.dumps(summary, indent=1))
