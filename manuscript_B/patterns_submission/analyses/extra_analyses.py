"""Exploratory analyses added before submission to Patterns, from the deposited records only (no model refits).

1. Corpus composition by repository and assay platform.
2. Pre-split null inflation and the observed gap across sample-size and feature-to-sample bins
   (design quantities, so they share no estimation noise with the gap).
3. The same by assay platform.
4. Stability of fold-contained feature selection (Nogueira index, mean Jaccard) and its association
   with the gap and with the fold-contained AUC.
5. Per-dataset values behind figure panels: shares of positive, zero and negative gaps; the 100-dataset
   null subset by arm; the two exploratory pipelines; permuted held-out labels; the human subset.

All intervals are dataset-level percentile bootstrap 95% CIs (2,000 resamples, seed 20261018). These
analyses were chosen after the formal results were known and are reported as exploratory (Table S12).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
MB = HERE.parents[1]
TABLES = MB / "Formal_Public_Deposit_v3" / "tables"
RES_BC = MB / "benchmark_analyses" / "control_arm" / "results"
SD1 = HERE.parent / "output" / "Patterns_Data_S1_per_dataset.csv"
OUT = HERE / "results"
METHODS = ["top_k", "rfe", "spls", "pls_da", "opls_da"]
SEED, B = 20261018, 2000
N_EDGES, N_LABELS = [30, 40, 60, 100, 200, np.inf], ["30–39", "40–59", "60–99", "100–199", "≥200"]
R_EDGES, R_LABELS = [0, 1, 10, 100, 1000, np.inf], ["<1", "1–10", "10–100", "100–1,000", "≥1,000"]
S_EDGES, S_LABELS = [-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf], ["<0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "≥0.8"]
GROUP = {"metabolomics": "Metabolomics", "MICROARRAY_1COLOUR_MRNA_DIFFERENTIAL": "Microarray",
         "MICROARRAY_1COLOUR_MICRORNA_DIFFERENTIAL": "Microarray", "RNASEQ_MRNA_DIFFERENTIAL": "RNA-seq",
         "RNASEQ_MRNA_BASELINE": "RNA-seq", "PROTEOMICS_BASELINE": "Proteomics", "microbiome": "Microbiome (16S)"}
GROUPS = ["Metabolomics", "Microarray", "RNA-seq", "Microbiome (16S)", "Proteomics"]


def med_ci(x) -> dict:
    x = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce").dropna(), dtype=float)
    if x.size == 0:
        return {"n": 0, "median": None, "ci": [None, None]}
    rng = np.random.default_rng(SEED)
    boots = np.median(rng.choice(x, (B, x.size)), axis=1)
    return {"n": int(x.size), "median": float(np.median(x)), "q25": float(np.percentile(x, 25)),
            "q75": float(np.percentile(x, 75)),
            "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(TABLES / "meta_input.csv", low_memory=False)
    for c in ("gap", "pre_split_auc", "fold_contained_auc", "n_samples", "n_features"):
        meta[c] = pd.to_numeric(meta[c], errors="coerce")
    meta["group"] = meta.assay_platform.map(GROUP)
    assert meta.group.notna().all(), "unmapped assay platform"
    meta["ratio"] = meta.n_features / meta.n_samples
    meta["n_bin"] = pd.cut(meta.n_samples, N_EDGES, right=False, labels=N_LABELS)
    meta["r_bin"] = pd.cut(meta.ratio, R_EDGES, right=False, labels=R_LABELS)
    ps = pd.read_csv(TABLES / "permutation_summary.csv", low_memory=False)
    ps = ps[(ps.is_formal == True) & ps.metric.isin(["pre_split_oof_auc", "fold_contained_oof_auc"])]  # noqa: E712
    nulls = ps.pivot_table(index=["dataset_id", "method"], columns="metric", values="null_mean").reset_index()
    nulls = nulls.rename(columns={"pre_split_oof_auc": "null_pre", "fold_contained_oof_auc": "null_fc"})
    nulls = nulls.merge(meta[["dataset_id", "method", "n_samples", "n_features", "ratio", "n_bin", "r_bin", "group",
                              "repository"]], on=["dataset_id", "method"], how="left", validate="one_to_one")
    return meta, nulls


def corpus(meta: pd.DataFrame) -> dict:
    d = meta.drop_duplicates("dataset_id")
    d[["dataset_id", "repository", "group", "n_samples", "n_features", "n_pos", "n_neg"]].to_csv(
        OUT / "corpus_per_dataset.csv", index=False)
    by = {}
    for g, x in d.groupby("group"):
        by[g] = {"n_datasets": int(len(x)), "median_n": float(x.n_samples.median()),
                 "median_p": float(x.n_features.median()),
                 "repositories": x.repository.value_counts().to_dict()}
    return {"n_datasets": int(len(d)), "median_n": float(d.n_samples.median()),
            "n_iqr": [float(d.n_samples.quantile(.25)), float(d.n_samples.quantile(.75))],
            "median_p": float(d.n_features.median()),
            "p_iqr": [float(d.n_features.quantile(.25)), float(d.n_features.quantile(.75))],
            "median_ratio": float(d.ratio.median()), "by_group": by,
            "by_repository": d.repository.value_counts().to_dict(),
            "n_bin_counts": d.n_bin.value_counts().reindex(N_LABELS).astype(int).to_dict(),
            "r_bin_counts": d.r_bin.value_counts().reindex(R_LABELS).astype(int).to_dict()}


def by_bins(df: pd.DataFrame, value: str, binc: str, labels: list[str]) -> dict:
    return {m: {lab: med_ci(df[(df.method == m) & (df[binc] == lab)][value]) for lab in labels} for m in METHODS}


def stability(meta: pd.DataFrame) -> dict:
    fs = pd.read_csv(TABLES / "feature_stability.csv.gz", usecols=["dataset_id", "method", "arm", "nogueira",
                                                                "mean_jaccard", "mean_q"])
    x = meta.merge(fs, on=["dataset_id", "method"], how="inner", validate="one_to_one")
    x["s_bin"] = pd.cut(x.nogueira, S_EDGES, right=False, labels=S_LABELS)
    x[["dataset_id", "method", "nogueira", "mean_jaccard", "mean_q", "gap", "fold_contained_auc",
       "n_samples"]].to_csv(OUT / "stability_per_dataset.csv", index=False)
    out = {}
    for m in METHODS:
        g = x[x.method == m]
        ok = g.nogueira.notna() & g.gap.notna()
        out[m] = {"n": int(ok.sum()), "nogueira": med_ci(g.nogueira), "jaccard": med_ci(g.mean_jaccard),
                  "rho_gap": float(spearmanr(g.nogueira[ok], g.gap[ok])[0]),
                  "rho_fc_auc": float(spearmanr(g.nogueira[ok], g.fold_contained_auc[ok])[0]),
                  "rho_n": float(spearmanr(g.nogueira[ok], g.n_samples[ok])[0]),
                  "rho_jaccard_gap": float(spearmanr(g.mean_jaccard[ok], g.gap[ok])[0]),
                  "gap_by_bin": {lab: med_ci(g[g.s_bin == lab].gap) for lab in S_LABELS}}
    return out


def gap_shares(meta: pd.DataFrame) -> dict:
    out = {}
    for m in METHODS:
        g = meta[meta.method == m].gap.dropna()
        out[m] = {"n": int(g.size), "positive": float((g > 0).mean()), "zero": float((g == 0).mean()),
                  "negative": float((g < 0).mean()), "below_minus_0.05": float((g < -0.05).mean()),
                  "above_0.10": float((g > 0.10).mean())}
    meta[["dataset_id", "method", "gap", "pre_split_auc", "fold_contained_auc"]].to_csv(OUT / "observed_gaps.csv",
                                                                                        index=False)
    return out


def _per_dataset_null(folder: str, inner: str, arms: list[str], keys: list[str]) -> pd.DataFrame:
    rows = []
    for f in sorted((RES_BC / folder).glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("status") != "ok":
            continue
        for k in keys:
            for arm in arms:
                vals = [p[inner][k][f"{arm}_oof_auc"] for p in r["permutations"].values()]
                rows.append({"dataset_id": r["dataset_id"], "key": k, "arm": arm, "n_perm": len(vals),
                             "null_mean": float(np.mean(vals))})
    return pd.DataFrame(rows)


def per_dataset_panels(meta: pd.DataFrame) -> dict:
    nc = _per_dataset_null("v2_nullcheck", "methods", ["fold_contained", "global_prep", "pre_split", "pre_tuned",
                                                        "swap"], METHODS)
    nc.to_csv(OUT / "nullcheck_per_dataset.csv", index=False)
    gen = _per_dataset_null("v4_generality_null", "pipelines", ["pre_split", "fold_contained"], ["vip", "rf"])
    gen.to_csv(OUT / "generality_null_per_dataset.csv", index=False)
    sd1 = pd.read_csv(SD1, low_memory=False)
    ph = []
    for m in METHODS:
        d = sd1[["dataset_id", f"{m}_permuted_heldout_auc", f"{m}_swap_auc", f"{m}_global_prep_auc",
                 f"{m}_fold_contained_auc", f"{m}_pre_split_auc"]].dropna()
        ph.append(pd.DataFrame({"dataset_id": d.dataset_id, "method": m,
                                "permuted_heldout_minus_fc": d[f"{m}_permuted_heldout_auc"] - d[f"{m}_fold_contained_auc"],
                                "swap_minus_fc": d[f"{m}_swap_auc"] - d[f"{m}_fold_contained_auc"],
                                "global_prep_minus_fc": d[f"{m}_global_prep_auc"] - d[f"{m}_fold_contained_auc"],
                                "gap": d[f"{m}_pre_split_auc"] - d[f"{m}_fold_contained_auc"]}))
    ph = pd.concat(ph)
    ph.to_csv(OUT / "decomposition_per_dataset.csv", index=False)
    grad = []
    for m in METHODS:
        cols = {"0.00": None, "0.25": f"{m}_gap_permuted_0.25", "0.50": f"{m}_gap_permuted_0.50",
                "0.75": f"{m}_gap_permuted_0.75"}
        d = sd1[sd1.primary_null_eligible == True]                                   # noqa: E712
        for _, r in d.iterrows():
            row = {"dataset_id": r.dataset_id, "method": m,
                   "0.00": r[f"{m}_pre_split_auc"] - r[f"{m}_fold_contained_auc"],
                   "1.00": r[f"{m}_pre_split_null_mean_auc"] - r[f"{m}_fold_contained_null_mean_auc"]}
            row.update({k: r[c] for k, c in cols.items() if c})
            grad.append(row)
    grad = pd.DataFrame(grad)
    grad.to_csv(OUT / "gradient_per_dataset.csv", index=False)
    return {"n_nullcheck_datasets": int(nc.dataset_id.nunique()),
            "n_generality_null_datasets": int(gen.dataset_id.nunique()),
            "gradient_share_increase_obs_to_null": {
                m: float((grad[grad.method == m]["1.00"] > grad[grad.method == m]["0.00"]).mean()) for m in METHODS}}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    meta, nulls = load()
    exch = nulls.dropna(subset=["null_pre", "null_fc"])
    flow = pd.read_csv(TABLES / "dataset_eligibility_flow.csv", low_memory=False)
    primary = set(flow.loc[flow.primary_null_eligible == True, "dataset_id"])          # noqa: E712
    exch = exch[exch.dataset_id.isin(primary)]
    assert exch.dataset_id.nunique() == 1030, exch.dataset_id.nunique()
    exch.to_csv(OUT / "nulls_by_design.csv", index=False)
    res = {"corpus": corpus(meta),
           "null_pre_by_n": by_bins(exch, "null_pre", "n_bin", N_LABELS),
           "null_fc_by_n": by_bins(exch, "null_fc", "n_bin", N_LABELS),
           "null_pre_by_ratio": by_bins(exch, "null_pre", "r_bin", R_LABELS),
           "null_fc_by_ratio": by_bins(exch, "null_fc", "r_bin", R_LABELS),
           "gap_by_n": by_bins(meta, "gap", "n_bin", N_LABELS),
           "gap_by_ratio": by_bins(meta, "gap", "r_bin", R_LABELS),
           "ceiling_share_by_ratio": {m: {lab: float((meta[(meta.method == m) & (meta.r_bin == lab)]
                                                      .fold_contained_auc >= 0.95).mean()) for lab in R_LABELS}
                                      for m in METHODS},
           "ceiling_share_by_n": {m: {lab: float((meta[(meta.method == m) & (meta.n_bin == lab)]
                                                  .fold_contained_auc >= 0.95).mean()) for lab in N_LABELS}
                                  for m in METHODS},
           "null_pre_by_group": {m: {g: med_ci(exch[(exch.method == m) & (exch.group == g)].null_pre) for g in GROUPS}
                                 for m in METHODS},
           "null_fc_by_group": {m: {g: med_ci(exch[(exch.method == m) & (exch.group == g)].null_fc) for g in GROUPS}
                                for m in METHODS},
           "gap_by_group": {m: {g: med_ci(meta[(meta.method == m) & (meta.group == g)].gap) for g in GROUPS}
                            for m in METHODS},
           "stability": stability(meta), "gap_shares": gap_shares(meta),
           "panels": per_dataset_panels(meta),
           "labels": {"n": N_LABELS, "ratio": R_LABELS, "stability": S_LABELS, "groups": GROUPS}}
    (OUT / "extra_results.json").write_text(json.dumps(res, indent=1))
    c = res["corpus"]
    print("corpus", c["n_datasets"], c["by_repository"], {g: v["n_datasets"] for g, v in c["by_group"].items()})
    for m in METHODS:
        print(m, "null_pre by n", [round(res["null_pre_by_n"][m][l]["median"], 3) for l in N_LABELS],
              "| by ratio", [round(res["null_pre_by_ratio"][m][l]["median"], 3) for l in R_LABELS])
        print("   gap by n", [round(res["gap_by_n"][m][l]["median"], 3) for l in N_LABELS],
              "| by ratio", [round(res["gap_by_ratio"][m][l]["median"], 3) for l in R_LABELS])
        print("   null_pre by group", {g: (round(v["median"], 3) if v["median"] is not None else None, v["n"])
                                        for g, v in res["null_pre_by_group"][m].items()})
        s = res["stability"][m]
        print("   stability nog %.2f jac %.2f rho_gap %.2f rho_fc %.2f rho_n %.2f" % (
            s["nogueira"]["median"], s["jaccard"]["median"], s["rho_gap"], s["rho_fc_auc"], s["rho_n"]),
              "gap by stab", [(round(v["median"], 3) if v["median"] is not None else None, v["n"])
                              for v in s["gap_by_bin"].values()])
    print("shares", {m: {k: round(v, 3) for k, v in d.items() if k != "n"} for m, d in res["gap_shares"].items()})
    print("panels", res["panels"])


if __name__ == "__main__":
    main()
