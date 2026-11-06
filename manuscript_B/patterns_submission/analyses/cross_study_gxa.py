"""Validation in other studies, Expression Atlas (exploratory; the transcriptomic and proteomic counterpart of
reserve/cross_cohort.py, which used MicrobiomeHD).

Study pairs. Among the analysed Expression Atlas datasets (gxa_survey.py), human contrasts studied by at least two
datasets on the same technology and tissue: rheumatoid factor-negative juvenile idiopathic arthritis against
controls (blood), Alzheimer's disease against controls (brain), psoriasis against controls (skin), lesional against
non-lesional psoriatic skin, atopic eczema lesions against normal skin, Alzheimer's against asymptomatic Alzheimer's
disease (brain proteomics) and breast cancer against normal breast (two array platforms, paired within platform).
Asthma (two lymphoblastoid datasets) was a candidate but overlapped (below). Rat toxicogenomic dose contrasts and a
cell-line compound contrast were not used: their datasets studied different compounds or cells.

Overlap. Two datasets of one contrast may contain the same samples. After centering each gene within each study, a
pair of studies counts as overlapping if most samples of either study are more similar (Pearson r) to some sample
of the other study than to any sample of their own study, with that cross-study match at r >= 0.5, or if they
share sample identifiers; of two overlapping datasets the larger is kept.

Analysis, for each ordered pair (training study A, external study B), exactly as in cross_cohort.py: the genes
measured in both studies; labels oriented so that 1 is the disease (or lesion) in both; the formal split plan for A
(group-aware where participants are recorded), the formal fold preprocessing for the data type, seed schedule and
five methods; one final model fitted to all of A (candidate chosen most often across the 15 outer folds; the
preprocessing fitted on A and applied unchanged to B) and scored once on B; 100 permutations of A's labels, each with
its own split plan and final model. Output: results/cross_study_gxa_records.json (checkpointed in
results/cross_study_gxa.partial.jsonl).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "validation"))
sys.path.insert(0, str(ROOT / "reserve"))
import validate_against_formal as V                       # noqa: E402

OUT = HERE / "results"
PARTIAL, RECORDS = OUT / "cross_study_gxa.partial.jsonl", OUT / "cross_study_gxa_records.json"
N_PERM = 100
# contrast: (reference level, datasets on one technology and tissue)
CONTRASTS = {
    "JIA": ("normal", ("E-GEOD-11083", "E-GEOD-13501", "E-GEOD-13849", "E-GEOD-41831")),
    "AD": ("normal", ("E-GEOD-48350", "E-GEOD-5281")),
    "Psoriasis": ("normal", ("E-GEOD-32407", "E-GEOD-52471")),
    "Psoriasis lesion": ("non-lesional", ("E-GEOD-14905", "E-GEOD-30999", "E-GEOD-41662")),
    "Eczema lesion": ("normal", ("E-GEOD-32924", "E-GEOD-36842", "E-GEOD-58558")),
    "AD proteome": ("asymptomatic Alzheimer's disease", ("E-PROT-51", "E-PROT-54")),
    "Breast cancer (Affymetrix)": ("normal", ("E-GEOD-10780", "E-GEOD-54002")),
    "Breast cancer (Agilent)": ("normal", ("E-MTAB-779", "E-GEOD-38959")),
    "Asthma": ("normal", ("E-GEOD-8052", "E-MTAB-1425")),
}
CLUSTER = {"Breast cancer (Affymetrix)": "Breast cancer", "Breast cancer (Agilent)": "Breast cancer"}


def load(dataset_id: str, reference: str):
    from benchmark_v4 import raw_loaders
    (platform, entry), = raw_loaders.entries_for_whitelist([dataset_id])
    config, ds, X, y = V._load(dataset_id, platform, dict(entry))
    level_pos, level_neg = ds.metadata["levels"]
    if reference not in (level_pos, level_neg):
        raise ValueError(f"{dataset_id}: reference level {reference!r} not in {level_pos!r}, {level_neg!r}")
    case = (1 - y) if level_pos == reference else y        # 1 = disease or lesion in every study
    frame = pd.DataFrame(np.asarray(X, dtype=float), index=list(ds.sample_ids), columns=list(ds.feature_ids))
    groups = None if ds.groups is None else np.asarray(ds.groups, dtype=object)
    if groups is not None and np.unique(groups).size == groups.size:
        groups = None
    return config, ds.transform_kind, frame, np.asarray(case, dtype=int), groups


def _centered(frame: pd.DataFrame, genes) -> np.ndarray:
    x = frame[genes].to_numpy()
    x = np.log2(np.clip(x, 0, None) + 1) if np.nanmax(x) > 50 else x
    x = x - x.mean(axis=0)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def overlap(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    genes = a.columns.intersection(b.columns)
    xa, xb = a[genes].to_numpy(), b[genes].to_numpy()
    keep = np.isfinite(xa).all(0) & np.isfinite(xb).all(0) & (xa.std(0) > 0) & (xb.std(0) > 0)
    genes = genes[keep]
    za, zb = _centered(a, genes), _centered(b, genes)
    cross = za @ zb.T
    out = {"shared_ids": len(set(a.index) & set(b.index))}
    for name, z, c in (("a", za, cross), ("b", zb, cross.T)):
        within = z @ z.T
        np.fill_diagonal(within, -np.inf)
        best_other, best_own = c.max(axis=1), within.max(axis=1)
        out[f"{name}_share_closer_elsewhere"] = float(np.mean(best_other > best_own))
        out[f"{name}_median_cross_max"] = float(np.median(best_other))
    out["overlapping"] = bool(out["shared_ids"] > 0 or any(
        out[f"{k}_share_closer_elsewhere"] > 0.5 and out[f"{k}_median_cross_max"] >= 0.5 for k in ("a", "b")))
    return out


def study_pairs() -> tuple[list[tuple[str, str, str]], dict]:
    """Ordered pairs after dropping the smaller dataset of every overlapping pair; with the overlap diagnostics."""
    diagnostics, pairs = {}, []
    for contrast, (reference, datasets) in CONTRASTS.items():
        data = {d: load(d, reference) for d in datasets}
        dropped = set()
        for x, z in itertools.combinations(datasets, 2):
            o = overlap(data[x][2], data[z][2])
            diagnostics[f"{contrast}|{x}|{z}"] = o
            if o["overlapping"]:
                dropped.add(min((x, z), key=lambda d: (data[d][3].size, d)))
        kept = [d for d in datasets if d not in dropped]
        pairs += [(contrast, a, b) for a, b in itertools.permutations(kept, 2)]
    return pairs, diagnostics


def run_pair(job) -> dict:
    import common as C
    contrast, a_name, b_name, n_perm = job
    t0 = time.time()
    reference = CONTRASTS[contrast][0]
    cfg, kind_a, fa, ya, groups_a = load(a_name, reference)
    _, kind_b, fb, yb, _ = load(b_name, reference)
    if kind_a != kind_b:
        raise ValueError(f"{a_name} and {b_name} have different data types ({kind_a}, {kind_b})")
    genes = fa.columns.intersection(fb.columns)
    genes = genes[np.isfinite(fa[genes].to_numpy()).all(0) & np.isfinite(fb[genes].to_numpy()).all(0)]
    Xa, Xb = fa[genes].to_numpy(), fb[genes].to_numpy()
    unit = f"xstudy|{contrast}|{a_name}"
    prep = C.preprocessor_for(cfg, kind_a)
    labels = [("observed", ya)] + [(i, C.permuted(ya, groups_a, unit, i, cfg)) for i in range(n_perm)]
    rec = {"contrast": contrast, "cluster": CLUSTER.get(contrast, contrast), "case": "1 = disease or lesion",
           "train": a_name, "test": b_name, "n_train": int(ya.size), "n_train_cases": int(ya.sum()),
           "n_test": int(yb.size), "n_test_cases": int(yb.sum()), "genes_train": int(fa.shape[1]),
           "genes_test": int(fb.shape[1]), "genera_shared": int(len(genes)),
           "train_groups": None if groups_a is None else int(np.unique(groups_a).size), "methods": {}}
    per = {m: [] for m in C.TASK_METHODS["binary"]}
    for tag, yy in labels:
        folds = C.binary_plan(yy, groups_a, unit, cfg)
        cache = {}
        for m in C.TASK_METHODS["binary"]:
            out = C.evaluate_generic(Xa, yy, folds, C.make_step("binary", m, cfg), preprocessor=prep, task="binary",
                                     seeds=C.V.formal_seeds(cfg, unit, m), cache=cache)
            scores, best = C.final_model_predictions(
                Xa, yy, Xb, C.make_step("binary", m, cfg), out["chosen_candidates"], preprocessor=prep,
                task="binary", rep_seed=C.derive_seed(cfg.seed, unit, m, "external_final"),
                clf_seed=C.derive_seed(cfg.seed, unit, m, "external_final_clf"))
            per[m].append({"labels": tag, "pre_split": out["pre_split"], "fold_contained": out["fold_contained"],
                           "external": float(roc_auc_score(yb, scores)), "final_candidate": int(best)})
    rec["outer_folds"] = int(len(C.binary_plan(ya, groups_a, unit, cfg)) // cfg.outer_repeats)
    rec["methods"] = {m: {"observed": v[0], "permuted": v[1:]} for m, v in per.items()}
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--permutations", type=int, default=N_PERM)
    args = ap.parse_args()
    if not all(V.verify_formal_code().values()):
        raise RuntimeError("formal code mismatch")
    pairs, diagnostics = study_pairs()
    (OUT / "cross_study_gxa_overlap.json").write_text(json.dumps(diagnostics, indent=1))
    done = set()
    if PARTIAL.exists():
        done = {(r["contrast"], r["train"], r["test"]) for r in map(json.loads, PARTIAL.read_text().splitlines())}
    todo = [p for p in pairs if p not in done]
    print(f"{len(pairs)} ordered pairs in {len({p[0] for p in pairs})} contrasts; {len(todo)} to run", flush=True)
    errors, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool, PARTIAL.open("a") as sink:
        futures = {pool.submit(run_pair, (*p, args.permutations)): p for p in todo}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                sink.write(json.dumps(fut.result()) + "\n")
                sink.flush()
            except Exception as exc:                       # recorded, never silently dropped
                errors.append({"pair": list(futures[fut]), "error": repr(exc)})
            print(f"  {k}/{len(todo)} {futures[fut][0]} {futures[fut][1]}->{futures[fut][2]}  "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
    if errors:
        (OUT / "cross_study_gxa.errors.json").write_text(json.dumps(errors, indent=1))
        print(f"{len(errors)} failed pairs; rerun to retry them", flush=True)
        return 1
    records = [json.loads(line) for line in PARTIAL.read_text().splitlines()]
    assert {(r["contrast"], r["train"], r["test"]) for r in records} == set(pairs)
    RECORDS.write_text(json.dumps(sorted(records, key=lambda r: (r["contrast"], r["train"], r["test"])), indent=1))
    print("written", RECORDS.name, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
