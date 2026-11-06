"""Cross-cohort external validation in MicrobiomeHD (exploratory; reported in the article, Figure 5E-F, Tables S21-S22).

Question. The hold-out analysis of the article (Figure 5C) withheld samples of the same study. Does its conclusion
hold when the external samples come from another study of the same disease, with its own population, sequencing
region and laboratory? For each ordered pair of studies of one contrast (training cohort A, external cohort B),
both protocols are cross-validated on A and one final model fitted on A is scored once on B. The protocols share
that final model, so each protocol's cross-validated AUC minus the external AUC is its error as an estimate of
performance in a new cohort; cohort shift lowers the external AUC for both protocols alike.

Contrasts. Disease against the formal healthy controls, for every disease with at least two MicrobiomeHD studies
having at least 10 samples per class: ASD, CDI, CRC, HIV, Crohn's disease, ulcerative colitis, NASH and obesity (8
contrasts, 26 study-contrasts, 66 ordered pairs). T1D is excluded (t1d_mejialeon has 8 controls). The case group
is named, so it can differ from the formal dataset of a study, which took the most frequent case group
(crc_baxter and crc_zackular: nonCRC; cdi_youngster: postFMT_CDI; nash_ob_baker: nonNASH-OB; ob_goodrich: OW).

Features. Read counts summed to named genera (RDP lineage down to genus), restricted to the genera detected in
both cohorts of the pair. The formal microbiome preprocessing (relative abundance over these genera, log2 with a
training pseudocount, completeness and variance filters, z-scoring) is fitted on A and applied unchanged to B.

Internal estimates. The formal split plan for A's labels (5 folds x 3 repeats, 3 inner folds; group-aware), the
formal seed schedule and the five formal methods (presplit-audit), pooled out-of-fold AUC. Folds and seeds depend
on the contrast and A only, so they are the same for every external partner.

Final model. As in the hold-out analysis: preprocessing, supervised step and classifier fitted on all of A, with
the candidate chosen most often across the 15 outer folds, scored once on B.

Null control. 100 permutations of A's labels, each with its own split plan and final model; B keeps its true
labels, so the external AUC of a permuted-label model has expectation 0.5. (A first run with 10 permutations,
whose permutations are the first 10 of these, is kept in results/archive_10perm.)

Output: results/cross_cohort_records.json, one record per ordered pair.
"""
from __future__ import annotations

import argparse
import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import roc_auc_score

import common as C

CONTRASTS = {
    "ASD": ("ASD", ("asd_son", "autism_kb")),
    "CDI": ("CDI", ("cdi_schubert", "cdi_vincent_v3v5", "cdi_youngster")),
    "CRC": ("CRC", ("crc_baxter", "crc_xiang", "crc_zackular", "crc_zeller", "crc_zhao")),
    "HIV": ("HIV", ("hiv_dinh", "hiv_lozupone", "hiv_noguerajulian")),
    "CD": ("CD", ("ibd_alm", "ibd_engstrand_maxee", "ibd_gevers_2014", "ibd_huttenhower")),
    "UC": ("UC", ("ibd_alm", "ibd_engstrand_maxee", "ibd_huttenhower")),
    "NASH": ("NASH", ("nash_chan", "nash_ob_baker")),
    "OB": ("OB", ("ob_goodrich", "ob_gordon_2008_v2", "ob_ross", "ob_zupancic")),
}
N_PERM, MIN_PER_CLASS = 100, 10
OUT = C.RESULTS / "cross_cohort_records.json"


def pairs() -> list[tuple[str, str, str, str]]:
    return [(contrast, case, a, b) for contrast, (case, studies) in CONTRASTS.items()
            for a, b in itertools.permutations(studies, 2)]


def cohort(study: C.Study, case: str):
    mask = np.isin(study.state, [case, "H"])
    y = (study.state[mask] == case).astype(int)
    if min(np.bincount(y, minlength=2)) < MIN_PER_CLASS:
        raise ValueError(f"{study.study}: fewer than {MIN_PER_CLASS} samples in a class")
    genera = C.genus_counts(study, mask)
    detected = set(genera.columns[(genera.to_numpy() > 0).any(axis=0)])
    return genera, y, None if study.groups is None else study.groups[mask], detected


def run_pair(job) -> dict:
    contrast, case, a_name, b_name, n_perm = job
    t0 = time.time()
    cfg = C.formal_config()
    ga, ya, groups_a, det_a = cohort(C.load_study(a_name), case)
    gb, yb, _, det_b = cohort(C.load_study(b_name), case)
    shared = sorted(det_a & det_b)
    Xa, Xb = ga[shared].to_numpy(), gb[shared].to_numpy()
    unit = f"xcohort|{contrast}|{a_name}"
    prep = C.preprocessor_for(cfg, "microbiome_counts")
    labels = [("observed", ya)] + [(i, C.permuted(ya, groups_a, unit, i, cfg)) for i in range(n_perm)]
    plans = {tag: C.binary_plan(yy, groups_a, unit, cfg) for tag, yy in labels}
    rec = {"contrast": contrast, "case": case, "train": a_name, "test": b_name, "n_train": int(ya.size),
           "n_train_cases": int(ya.sum()), "n_test": int(yb.size), "n_test_cases": int(yb.sum()),
           "genera_train": len(det_a), "genera_test": len(det_b), "genera_shared": len(shared),
           "train_groups": None if groups_a is None else int(np.unique(groups_a).size),
           "outer_folds": int(len(plans["observed"]) // cfg.outer_repeats), "methods": {}}
    for m in C.TASK_METHODS["binary"]:
        seeds = C.V.formal_seeds(cfg, unit, m)
        per = []
        for tag, yy in labels:
            out = C.evaluate_generic(Xa, yy, plans[tag], C.make_step("binary", m, cfg), preprocessor=prep,
                                     task="binary", seeds=seeds)
            scores, best = C.final_model_predictions(
                Xa, yy, Xb, C.make_step("binary", m, cfg), out["chosen_candidates"], preprocessor=prep,
                task="binary", rep_seed=C.derive_seed(cfg.seed, unit, m, "external_final"),
                clf_seed=C.derive_seed(cfg.seed, unit, m, "external_final_clf"))
            per.append({"labels": tag, "pre_split": out["pre_split"], "fold_contained": out["fold_contained"],
                        "external": float(roc_auc_score(yb, scores)), "final_candidate": int(best)})
        rec["methods"][m] = {"observed": per[0], "permuted": per[1:]}
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--permutations", type=int, default=N_PERM)
    ap.add_argument("--pilot", nargs="*", default=None, help="contrast:train:test triples to run alone")
    args = ap.parse_args()
    C.check_formal_code()
    jobs = [(*p, args.permutations) for p in pairs()]
    if args.pilot is not None:
        wanted = {tuple(s.split(":")) for s in args.pilot}
        jobs = [j for j in jobs if (j[0], j[2], j[3]) in wanted]
    assert len(pairs()) == 66
    print(f"{len(jobs)} ordered pairs, {args.permutations} permutations each", flush=True)
    records, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_pair, j): j for j in jobs}
        for k, fut in enumerate(as_completed(futures), 1):
            j = futures[fut]
            try:
                records.append(fut.result())
            except Exception as exc:                       # recorded, never silently dropped
                records.append({"contrast": j[0], "train": j[2], "test": j[3], "error": repr(exc)})
            print(f"  {k}/{len(jobs)} {j[0]} {j[2]}->{j[3]}  {(time.time() - t0) / 60:.1f} min", flush=True)
    records.sort(key=lambda r: (r["contrast"], r["train"], r["test"]))
    path = OUT if args.pilot is None else OUT.with_name("cross_cohort_pilot.json")
    C.write_json(path, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
