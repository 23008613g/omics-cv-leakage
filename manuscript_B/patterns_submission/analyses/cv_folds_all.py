"""Ten against five outer folds in all 1,086 analysed datasets (extension of cv_folds_sensitivity.py; exploratory).

Same design as the 100-dataset run: 10 outer folds instead of 5, everything else formal (three repeats, three inner
folds, fold preprocessing, candidate grids, classifier, seed schedule) and the formal permuted label vectors 0-19.
To keep the largest datasets from setting the run time, each dataset is split into jobs of one method and one block
of label vectors; every label vector is evaluated exactly as in cv_folds_sensitivity.run_one (the seeds depend on the
dataset and method only), so the split leaves every value unchanged, which ``--verify`` checks against the 100-dataset
records. Those 100 datasets are taken from that run rather than recomputed. Finished jobs are appended to
``results/cv_folds_all.partial.jsonl`` and a rerun resumes from it; the assembled records go to
``results/cv_folds_all_records.json`` in the format of ``cv_folds_records.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "validation"))
import validate_against_formal as V                       # noqa: E402

OUT = HERE / "results"
DATA_S1 = ROOT / "output" / "Patterns_Data_S1_per_dataset.csv"
PARTIAL, RECORDS = OUT / "cv_folds_all.partial.jsonl", OUT / "cv_folds_all_records.json"
PARTIAL_LARGE = OUT / "cv_folds_all_large.partial.jsonl"   # written by a concurrent run of the largest jobs
CLAIMS = OUT / "cv_folds_all_claims"                       # with --claims, one file per job taken by a run


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def clean_claims() -> None:
    """Drop the claims of runs that no longer exist, so that their unfinished jobs can be taken again."""
    CLAIMS.mkdir(exist_ok=True)
    for f in CLAIMS.iterdir():
        if f.name.startswith("."):
            continue
        try:
            pid = int(f.read_text())
        except ValueError:
            continue
        if not _alive(pid):
            f.unlink(missing_ok=True)


def claim(job) -> bool:
    """Take a job for this run; False if another run has taken it. The claim is written in full before it is
    linked into place, so other runs never read a partial claim."""
    path = CLAIMS / f"{job[0]}|{job[4]}"
    tmp = CLAIMS / f".{os.getpid()}.{uuid.uuid4().hex}"
    tmp.write_text(str(os.getpid()))
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    finally:
        tmp.unlink(missing_ok=True)
SUBSET = OUT / "cv_folds_records.json"
N_PERM, OUTER = 20, 10
BLOCKS = (("observed", 0, 1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12, 13), (14, 15, 16, 17, 18, 19))


def analysed() -> pd.DataFrame:
    """The 1,086 analysed datasets: rows of Data S1 with formal AUCs."""
    s1 = pd.read_csv(DATA_S1)
    return s1[s1.top_k_pre_split_auc.notna()].reset_index(drop=True)


def evaluate_block(job) -> dict:
    """One method and one block of label vectors of one dataset. ``groups`` replaces the formal groups when given
    (family_sensitivity.py); ``outer`` is the number of outer folds."""
    from benchmark_v4.protocol import materialize_permuted_labels
    from benchmark_v4.splitting import materialize_plan

    dataset_id, platform, entry, method, block, outer, groups = job
    t0 = time.time()
    config, ds, X, y = V._load(dataset_id, platform, entry)
    if groups is None:
        groups = ds.groups
    split_config = replace(config, outer_splits=outer)
    prep = lambda: V.FormalPreprocessor(config, ds.transform_kind)       # noqa: E731
    seeds = V.formal_seeds(config, dataset_id, method)
    values, folds_used, units = [], set(), set()
    for tag in BLOCKS[block]:
        if tag == "observed":
            yy = y
        else:
            yy, unit = materialize_permuted_labels(y, config, dataset_id, tag, groups=groups)
            units.add(unit)
        folds = V.formal_folds(materialize_plan(yy, split_config, dataset_id, groups=groups))
        folds_used.add(len(folds) // split_config.outer_repeats)
        r = V.evaluate_protocols(X, yy, folds, V.make_step(method, config), preprocessor=prep, seeds=seeds)
        values.append({"labels": tag, "pre_split": r["pre_split_auc"], "fold_contained": r["fold_contained_auc"]})
    return {"dataset_id": dataset_id, "method": method, "block": block, "n": int(y.size), "p": int(X.shape[1]),
            "outer_folds": sorted(folds_used), "permutation_units": sorted(units), "values": values,
            "seconds": round(time.time() - t0, 1)}


def evaluate_combined(job) -> list[dict]:
    """All requested methods of one block of label vectors of one dataset, sharing the label-free fold
    preprocessing among the methods. Profiling showed that preprocessing takes 70-95% of a job on high-dimensional
    data and that each method repeated it. reserve/common.evaluate_generic caches it by the rows of each split and
    reproduces presplit-audit's evaluate_protocols exactly (reserve/tests; ``--verify-combined``). One record per
    method, in the format of evaluate_block."""
    from benchmark_v4.protocol import materialize_permuted_labels
    from benchmark_v4.splitting import materialize_plan
    sys.path.insert(0, str(ROOT / "reserve"))
    import common as C

    dataset_id, platform, entry, methods, block, outer = job
    t0 = time.time()
    config, ds, X, y = V._load(dataset_id, platform, entry)
    split_config = replace(config, outer_splits=outer)
    prep = lambda: V.FormalPreprocessor(config, ds.transform_kind)       # noqa: E731
    values, folds_used, units = {m: [] for m in methods}, set(), set()
    for tag in BLOCKS[block]:
        if tag == "observed":
            yy = y
        else:
            yy, unit = materialize_permuted_labels(y, config, dataset_id, tag, groups=ds.groups)
            units.add(unit)
        folds = V.formal_folds(materialize_plan(yy, split_config, dataset_id, groups=ds.groups))
        folds_used.add(len(folds) // split_config.outer_repeats)
        cache = {}                                         # one split plan per label vector, shared by the methods
        for m in methods:
            r = C.evaluate_generic(X, yy, folds, C.make_step("binary", m, config), preprocessor=prep, task="binary",
                                   seeds=V.formal_seeds(config, dataset_id, m), cache=cache)
            values[m].append({"labels": tag, "pre_split": r["pre_split"], "fold_contained": r["fold_contained"]})
    seconds = round((time.time() - t0) / len(methods), 1)
    return [{"dataset_id": dataset_id, "method": m, "block": block, "n": int(y.size), "p": int(X.shape[1]),
             "outer_folds": sorted(folds_used), "permutation_units": sorted(units), "values": values[m],
             "seconds": seconds, "combined": True} for m in methods]


def assemble(dataset_id: str, parts: list[dict]) -> dict:
    """Records in the format of cv_folds_records.json, with the per-permutation values kept."""
    first = parts[0]
    rec = {"dataset_id": dataset_id, "n": first["n"], "p": first["p"],
           "outer_folds": int(min(f for p in parts for f in p["outer_folds"])), "methods": {},
           "seconds": round(sum(p["seconds"] for p in parts), 1)}
    for m in V.METHODS:
        vals = {v["labels"]: v for p in parts if p["method"] == m for v in p["values"]}
        expected = ["observed"] + list(range(N_PERM))
        if sorted(vals, key=str) != sorted(expected, key=str):
            raise ValueError(f"{dataset_id}/{m}: label vectors {sorted(vals, key=str)}")
        perm = [vals[i] for i in range(N_PERM)]
        rec["methods"][m] = {"pre_split": vals["observed"]["pre_split"],
                             "fold_contained": vals["observed"]["fold_contained"],
                             "null_pre_split": float(np.mean([v["pre_split"] for v in perm])),
                             "null_fold_contained": float(np.mean([v["fold_contained"] for v in perm])),
                             "permuted": [{"labels": v["labels"], "pre_split": v["pre_split"],
                                           "fold_contained": v["fold_contained"]} for v in perm]}
    return rec


def memory_gb_combined(n: float, p: float) -> float:
    """Peak footprint of a combined job: the matrix and its temporaries plus the cache of the 121 preprocessed
    splits of one label vector (preprocessing keeps at most 5,000 features)."""
    return 0.3 + 3 * n * p * 8 / 1e9 + 121 * 1.15 * n * np.minimum(p, 5000) * 8 / 1e9


def memory_gb(n: float, p: float) -> float:
    """Approximate peak footprint of one job: about 15 copies of the float64 matrix plus the interpreter (measured
    peaks: 5 GB for 974 x 55,573 with ten folds, 6-8 GB for 786 x 72,256 with five family-aware folds)."""
    return 0.3 + 15 * n * p * 8 / 1e9


def verify(ids: list[str], workers: int) -> int:
    """The split jobs must reproduce the 100-dataset records exactly."""
    from benchmark_v4 import raw_loaders
    subset = {r["dataset_id"]: r for r in json.loads(SUBSET.read_text())}
    jobs = [(i, p, dict(e), m, b, OUTER, None) for i, (p, e) in zip(ids, raw_loaders.entries_for_whitelist(ids))
            for m in V.METHODS for b in range(len(BLOCKS))]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(evaluate_block, jobs))
    bad = 0
    for i in ids:
        rec = assemble(i, [p for p in parts if p["dataset_id"] == i])
        for m in V.METHODS:
            for k in ("pre_split", "fold_contained", "null_pre_split", "null_fold_contained"):
                bad += rec["methods"][m][k] != subset[i]["methods"][m][k]
        print(i, "identical" if bad == 0 else f"{bad} values differ", flush=True)
    return int(bad > 0)


def verify_combined(ids: list[str], workers: int) -> int:
    """The combined jobs must reproduce the 100-dataset records and the per-method jobs already checkpointed."""
    from benchmark_v4 import raw_loaders
    subset = {r["dataset_id"]: r for r in json.loads(SUBSET.read_text())}
    jobs = [(i, p, dict(e), list(V.METHODS), b, OUTER) for i, (p, e) in zip(ids, raw_loaders.entries_for_whitelist(ids))
            for b in range(len(BLOCKS))]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = [r for out in pool.map(evaluate_combined, jobs) for r in out]
    bad = 0
    for i in ids:
        rec = assemble(i, [p for p in parts if p["dataset_id"] == i])
        diff = sum(rec["methods"][m][k] != subset[i]["methods"][m][k] for m in V.METHODS
                   for k in ("pre_split", "fold_contained", "null_pre_split", "null_fold_contained"))
        bad += diff
        print(i, "identical" if diff == 0 else f"{diff} values differ", flush=True)
    return int(bad > 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--verify", nargs="*", default=None, help="datasets of the 100-dataset run to recompute")
    ap.add_argument("--memory-gb", type=float, default=22.0,
                    help="budget for the estimated footprints of the running jobs; large jobs wait for room while "
                         "small ones fill the other workers (too many large jobs at once made macOS compress memory "
                         "and stalled the workers)")
    ap.add_argument("--verify-combined", nargs="*", default=None,
                    help="datasets of the 100-dataset run to recompute with the combined jobs")
    ap.add_argument("--max-job-gb", type=float, default=None,
                    help="leave jobs estimated above this for a later run (their datasets stay unfinished)")
    ap.add_argument("--claims", action="store_true",
                    help="coordinate with other runs started with --claims: each job is claimed before it starts and "
                         "each run keeps its own checkpoint file, so more runs can be added while others go on")
    ap.add_argument("--only-large", type=float, default=None,
                    help="run only the jobs estimated above this many GB, checkpointed in a separate file, so that "
                         "this run can go alongside one started with the same value as --max-job-gb")
    args = ap.parse_args()
    code = V.verify_formal_code()
    if not all(code.values()):
        raise RuntimeError("formal code mismatch")
    if args.verify is not None:
        return verify(args.verify, args.workers)
    if args.verify_combined is not None:
        return verify_combined(args.verify_combined, args.workers)
    from benchmark_v4 import raw_loaders
    table = analysed()
    assert len(table) == 1086
    subset = {r["dataset_id"]: r for r in json.loads(SUBSET.read_text()) if "error" not in r}
    assert len(subset) == 100 and set(subset) <= set(table.dataset_id)
    def checkpointed() -> list[dict]:
        return [json.loads(line) for f in sorted(OUT.glob("cv_folds_all*.partial.jsonl"))
                for line in f.read_text().splitlines()]

    if args.claims:
        clean_claims()

    done = {(p["dataset_id"], p["method"], p["block"]) for p in checkpointed()}
    todo_ids = table[~table.dataset_id.isin(list(subset))].copy()
    # largest first, so that their jobs are spread over the workers early: liblinear and PLS costs grow with n² p
    todo_ids["cost"] = todo_ids.n_samples ** 2 * np.minimum(todo_ids.n_features, 5000)
    todo_ids = todo_ids.sort_values("cost", ascending=False)
    ids = list(todo_ids.dataset_id)
    resolved = dict(zip(ids, raw_loaders.entries_for_whitelist(ids), strict=True))
    size = todo_ids.set_index("dataset_id")
    # one job per dataset and block, with the methods not yet checkpointed (combined jobs share preprocessing)
    pending = []
    for i in ids:
        for b in range(len(BLOCKS)):
            methods = [m for m in V.METHODS if (i, m, b) not in done]
            if methods:
                pending.append(((i, resolved[i][0], dict(resolved[i][1]), methods, b, OUTER),
                                memory_gb_combined(size.loc[i, "n_samples"], size.loc[i, "n_features"])))
    if args.max_job_gb is not None:
        deferred = sum(mem > args.max_job_gb for _, mem in pending)
        pending = [(job, mem) for job, mem in pending if mem <= args.max_job_gb]
        print(f"{deferred} jobs above {args.max_job_gb:g} GB left for a later run", flush=True)
    if args.only_large is not None:
        pending = [(job, mem) for job, mem in pending if mem > args.only_large]
        print(f"running only the {len(pending)} jobs above {args.only_large:g} GB", flush=True)
    n_jobs = len(pending)
    print(f"{len(ids)} datasets to compute ({len(subset)} taken from the 100-dataset run); {n_jobs} jobs to run, "
          f"{len(done)} already in the checkpoint; memory budget {args.memory_gb:g} GB", flush=True)
    errors, t0, k, used, running = [], time.time(), 0, 0.0, {}
    if args.claims:
        sink_path = OUT / f"cv_folds_all_large.{os.getpid()}.partial.jsonl"
    else:
        sink_path = PARTIAL_LARGE if args.only_large is not None else PARTIAL
    with ProcessPoolExecutor(max_workers=args.workers) as pool, sink_path.open("a") as sink:
        while pending or running:
            # largest first, as memory allows: the first pending job that fits; always at least one job
            while pending and len(running) < args.workers:
                pick = next((j for j, (_, mem) in enumerate(pending) if used + mem <= args.memory_gb or not running),
                            None)
                if pick is None:
                    break
                job, mem = pending.pop(pick)
                if args.claims and not claim(job):
                    continue                               # another run has taken or finished it
                running[pool.submit(evaluate_combined, job)] = (job, mem)
                used += mem
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                job, mem = running.pop(fut)
                used -= mem
                k += 1
                try:
                    for rec in fut.result():
                        sink.write(json.dumps(rec) + "\n")
                    sink.flush()
                except Exception as exc:                   # recorded, never silently dropped
                    errors.append({"dataset_id": job[0], "methods": job[3], "block": job[4], "error": repr(exc)})
                    if args.claims:
                        (CLAIMS / f"{job[0]}|{job[4]}").unlink(missing_ok=True)
                if k % 50 == 0 or k == n_jobs:
                    print(f"  {k}/{n_jobs}  {(time.time() - t0) / 3600:.2f} h  errors {len(errors)}  running "
                          f"{len(running)}, {used:.1f} GB estimated", flush=True)
    if errors:
        (OUT / "cv_folds_all.errors.json").write_text(json.dumps(errors, indent=1))
        print(f"{len(errors)} failed jobs; rerun to retry them", flush=True)
        return 1
    parts = checkpointed()
    have = {(p["dataset_id"], p["method"], p["block"]) for p in parts}
    missing = sum((i, m, b) not in have for i in ids for m in V.METHODS for b in range(len(BLOCKS)))
    if missing:
        print(f"this run is finished; {missing} method-blocks are still missing (another run, or a rerun without "
              "--max-job-gb), so the records are not assembled yet", flush=True)
        return 0
    records = [subset[i] | {"source": "cv_folds_records.json"} for i in sorted(subset)]
    records += [assemble(i, [p for p in parts if p["dataset_id"] == i]) for i in ids]
    records.sort(key=lambda r: r["dataset_id"])
    assert len(records) == 1086
    RECORDS.write_text(json.dumps(records, indent=1))
    print("written", RECORDS.name, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
