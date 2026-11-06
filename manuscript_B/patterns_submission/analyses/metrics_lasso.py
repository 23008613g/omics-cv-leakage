"""Other metrics and a LASSO pipeline (exploratory).

1. Accuracy and balanced accuracy. The formal pipelines are rerun (formal five-fold plans, fold preprocessing,
   candidates, classifier and seed schedule) and the pooled out-of-fold predicted probabilities, averaged over the
   three repeats as for the AUC, are scored at a threshold of 0.5. The AUCs of the rerun must equal the formal ones,
   which shows that the new metrics come from the same pipelines.
2. LASSO selection. L1-penalized logistic regression (liblinear) on the preprocessed training data keeps the
   features with nonzero coefficients; the penalty C in {0.03, 0.1, 0.3, 1} is tuned in the inner folds like the
   other methods' candidates, and the classifier is the formal L2 logistic regression. If no coefficient is
   nonzero, the feature with the largest univariate F statistic is kept. Both protocols as for the formal methods.

Labels: the observed ones in all 1,086 analysed datasets, and the formal permuted label vectors 0-19 in the
100-dataset null subset (Table S3). The methods of one label vector share the label-free fold preprocessing.
Finished jobs are appended to results/metrics_lasso.partial.jsonl, and a rerun resumes; the records go to
results/metrics_lasso_records.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "validation"))
sys.path.insert(0, str(ROOT / "reserve"))
import validate_against_formal as V                       # noqa: E402
from cv_folds_all import analysed, memory_gb_combined     # noqa: E402

OUT = HERE / "results"
PARTIAL, RECORDS = OUT / "metrics_lasso.partial.jsonl", OUT / "metrics_lasso_records.json"
N_PERM = 20
BLOCKS = (("observed", 0, 1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12, 13), (14, 15, 16, 17, 18, 19))
METHODS = tuple(V.METHODS) + ("lasso",)


class LassoSelect:
    """Features with nonzero L1-logistic coefficients; one candidate per penalty."""

    name = "lasso"

    def __init__(self, penalties=(0.03, 0.1, 0.3, 1.0), max_iter: int = 5000):
        self.candidates = tuple({"C": float(c)} for c in penalties)
        self.max_iter = max_iter

    def fit_bundle(self, X, y, seed=0):
        out = []
        f_stat = None
        for cand in self.candidates:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model = LogisticRegression(penalty="l1", solver="liblinear", C=cand["C"], max_iter=self.max_iter,
                                           random_state=seed).fit(X, y)
            keep = np.flatnonzero(model.coef_.ravel() != 0)
            if keep.size == 0:
                if f_stat is None:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        f_stat = np.nan_to_num(f_classif(X, y)[0], nan=-np.inf)
                keep = np.array([int(np.argmax(f_stat))])
            out.append(lambda Z, keep=keep: np.asarray(Z)[:, keep])
        return out


def metrics(y, scores) -> dict:
    pred = (np.asarray(scores) >= 0.5).astype(int)
    return {"auc": float(roc_auc_score(y, scores)), "accuracy": float(np.mean(pred == y)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred))}


def job_fn(job) -> dict:
    import common as C
    from benchmark_v4.protocol import materialize_permuted_labels
    from benchmark_v4.splitting import materialize_plan
    dataset_id, platform, entry, tags, block = job
    t0 = time.time()
    config, ds, X, y = V._load(dataset_id, platform, entry)
    prep = lambda: V.FormalPreprocessor(config, ds.transform_kind)       # noqa: E731
    values = []
    for tag in tags:
        yy = y if tag == "observed" else materialize_permuted_labels(y, config, dataset_id, tag, groups=ds.groups)[0]
        folds = V.formal_folds(materialize_plan(yy, config, dataset_id, groups=ds.groups))
        cache, row = {}, {"labels": tag}
        for m in METHODS:
            step = LassoSelect() if m == "lasso" else C.make_step("binary", m, config)
            r = C.evaluate_generic(X, yy, folds, step, preprocessor=prep, task="binary",
                                   seeds=V.formal_seeds(config, dataset_id, m), cache=cache, return_scores=True)
            row[m] = {arm: metrics(yy, r["scores"][arm]) for arm in ("pre_split", "fold_contained")}
            assert row[m]["pre_split"]["auc"] == r["pre_split"] and row[m]["fold_contained"]["auc"] == r["fold_contained"]
        values.append(row)
    return {"dataset_id": dataset_id, "block": block, "n": int(y.size), "values": values,
            "seconds": round(time.time() - t0, 1)}


def formal_aucs(dataset_id: str) -> dict:
    ck = V.checkpoint(dataset_id)
    out = {("observed", m): (ck["observed"]["methods"][m]["observed"]["pre_split_oof_auc"],
                             ck["observed"]["methods"][m]["observed"]["fold_contained_oof_auc"]) for m in V.METHODS}
    for block in ck["permutation_blocks"]:
        res = block["result"]
        for m in V.METHODS:
            for i, a, b in zip(res["permutation_indices"], res["methods"][m]["pre_split_oof_auc"],
                               res["methods"][m]["fold_contained_oof_auc"], strict=True):
                out[(int(i), m)] = (a, b)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--memory-gb", type=float, default=30.0)
    args = ap.parse_args()
    if not all(V.verify_formal_code().values()):
        raise RuntimeError("formal code mismatch")
    from benchmark_v4 import raw_loaders
    table = analysed()
    subset = set(pd.read_csv(OUT / "nullcheck_per_dataset.csv").dataset_id)
    assert len(table) == 1086 and len(subset) == 100
    done = set()
    if PARTIAL.exists():
        done = {(p["dataset_id"], p["block"]) for p in map(json.loads, PARTIAL.read_text().splitlines())}
    table["cost"] = table.n_samples ** 2 * np.minimum(table.n_features, 5000)
    table = table.sort_values("cost", ascending=False)
    ids = list(table.dataset_id)
    resolved = dict(zip(ids, raw_loaders.entries_for_whitelist(ids), strict=True))
    size = table.set_index("dataset_id")
    pending = []
    for i in ids:
        blocks = list(enumerate(BLOCKS)) if i in subset else [(0, ("observed",))]
        mem = memory_gb_combined(size.loc[i, "n_samples"], size.loc[i, "n_features"]) / 2   # five folds, not ten
        pending += [((i, resolved[i][0], dict(resolved[i][1]), tags, b), mem) for b, tags in blocks if (i, b) not in done]
    n_jobs = len(pending)
    print(f"{n_jobs} jobs to run, {len(done)} in the checkpoint", flush=True)
    errors, k, used, running, t0 = [], 0, 0.0, {}, time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool, PARTIAL.open("a") as sink:
        while pending or running:
            while pending and len(running) < args.workers:
                pick = next((j for j, (_, mem) in enumerate(pending) if used + mem <= args.memory_gb or not running),
                            None)
                if pick is None:
                    break
                job, mem = pending.pop(pick)
                running[pool.submit(job_fn, job)] = (job, mem)
                used += mem
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                job, mem = running.pop(fut)
                used -= mem
                k += 1
                try:
                    sink.write(json.dumps(fut.result()) + "\n")
                    sink.flush()
                except Exception as exc:                   # recorded, never silently dropped
                    errors.append({"dataset_id": job[0], "block": job[4], "error": repr(exc)})
                if k % 50 == 0 or k == n_jobs:
                    print(f"  {k}/{n_jobs}  {(time.time() - t0) / 60:.1f} min  errors {len(errors)}", flush=True)
    if errors:
        (OUT / "metrics_lasso.errors.json").write_text(json.dumps(errors, indent=1))
        print(f"{len(errors)} failed jobs; rerun to retry them", flush=True)
        return 1
    parts = [json.loads(line) for line in PARTIAL.read_text().splitlines()]
    records, mismatches = {}, 0
    for p in parts:
        rec = records.setdefault(p["dataset_id"], {"dataset_id": p["dataset_id"], "n": p["n"], "values": []})
        rec["values"] += p["values"]
    for d, rec in records.items():
        expected = ["observed"] + (list(range(N_PERM)) if d in subset else [])
        assert sorted((v["labels"] for v in rec["values"]), key=str) == sorted(expected, key=str), d
        formal = formal_aucs(d)
        for v in rec["values"]:
            for m in V.METHODS:
                a, b = formal[(v["labels"], m)]
                mismatches += (v[m]["pre_split"]["auc"] != a) + (v[m]["fold_contained"]["auc"] != b)
    RECORDS.write_text(json.dumps({"mismatches_with_formal_auc": mismatches,
                                   "records": [records[d] for d in sorted(records)]}, indent=1))
    print(f"written {RECORDS.name}; AUC values differing from the formal run: {mismatches}", flush=True)
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
