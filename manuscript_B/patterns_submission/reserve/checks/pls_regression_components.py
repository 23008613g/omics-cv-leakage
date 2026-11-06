"""Check: why is the pre-split PLS R² under permuted outcomes so spread out (0.10-0.84 in ob_goodrich)?

Hypothesis: the candidate (number of PLS components, 1, 2, 3 or 5) is tuned fold-contained and shared by both
arms, as in the formal design. Under weak or no signal that choice is close to arbitrary, and the pre-split
leak grows with the number of components, so the pre-split R² depends on which count the honest tuning chose.

For ob_goodrich age and BMI: the observed outcome and the two permutations with the lowest and the two with
the highest pre-split R² in the main run. For each: (1) the main-run evaluation is repeated, which must
reproduce the stored values exactly, and the chosen candidates are recorded; (2) the pre-split R² with the
component count fixed at each candidate, from the same pre-split PLS fit (components are nested) and the same
ridge models, folds and seeds. Output: results/checks_pls_regression.json.
"""
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import common as C                                         # noqa: E402
import other_tasks as O                                    # noqa: E402

UNITS = (("ob_goodrich", "age", "age"), ("ob_goodrich", "body_mass_index", "bmi"))
METHOD = "pls_da"


def stored() -> dict:
    return {u["key"]: [u["methods"][METHOD]["observed"]] + u["methods"][METHOD]["permuted"] for u in
            json.loads((C.RESULTS / "other_tasks_real_regression.json").read_text())}


def picks(entries: list) -> list:
    perm = sorted((e for e in entries if e["labels"] != "observed"), key=lambda e: e["pre_split"])
    return ["observed"] + [e["labels"] for e in perm[:2] + perm[-2:]]


def one(job) -> dict:
    (study_name, column, kind), tag, expected = job
    cfg = C.formal_config()
    study = C.load_study(study_name)
    value = pd.to_numeric(study.metadata[column], errors="coerce").to_numpy(dtype=float)
    low, high = O.VALID[kind]
    mask = np.isfinite(value) & (value >= low) & (value <= high)
    y0 = value[mask]
    X, groups = study.rows(mask)
    unit = f"regression|{study_name}|{column}"
    folds = C.generic_plan(y0, groups, unit, cfg, stratify=False)
    (_, y), = O._vectors(y0, groups, unit, cfg, [tag])
    step, seeds = C.make_step("regression", METHOD, cfg), C.V.formal_seeds(cfg, unit, METHOD)
    prep = C.preprocessor_for(cfg, "microbiome_counts")
    cache = {}
    r = C.evaluate_generic(X, y, folds, step, preprocessor=prep, task="regression", seeds=seeds, cache=cache)
    fit_predict, score = C.MODELS["regression"]
    every = np.arange(y.size)
    gx = C._prepared(X, every, every, prep, cache)[0]
    global_z = [t(gx) for t in step.fit_bundle(gx, y, seed=seeds("global"))]
    fixed = {}
    for c, cand in enumerate(step.candidates):
        preds = [fit_predict(seeds("pre_split_classifier", o), global_z[c][f.train], y[f.train], global_z[c][f.test])
                 for o, f in enumerate(folds)]
        fixed[str(cand["n_components"])] = C.pooled(y, folds, preds, score)
    counts = Counter(step.candidates[c]["n_components"] for c in r["chosen_candidates"])
    return {"unit": f"{study_name}/{column}", "labels": tag, "stored_pre_split": expected["pre_split"],
            "stored_fold_contained": expected["fold_contained"], "pre_split": r["pre_split"],
            "fold_contained": r["fold_contained"],
            "identical": r["pre_split"] == expected["pre_split"] and r["fold_contained"] == expected["fold_contained"],
            "chosen_components": {str(k): v for k, v in sorted(counts.items())}, "pre_split_fixed_components": fixed}


def main() -> int:
    C.check_formal_code()
    S = stored()
    jobs = []
    for study_name, column, kind in UNITS:
        entries = S[f"{study_name}/{column}"]
        by_tag = {e["labels"]: e for e in entries}
        jobs += [((study_name, column, kind), tag, by_tag[tag]) for tag in picks(entries)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(one, jobs))
    for r in rows:
        print(f"{r['unit']:28s} {str(r['labels']):9s} pre {r['pre_split']:+.3f} fc {r['fold_contained']:+.3f} "
              f"identical {r['identical']}  chosen {r['chosen_components']}  fixed "
              + " ".join(f"k={k}:{v:+.2f}" for k, v in r["pre_split_fixed_components"].items()), flush=True)
    C.write_json(C.RESULTS / "checks_pls_regression.json", rows)
    return 0 if all(r["identical"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
