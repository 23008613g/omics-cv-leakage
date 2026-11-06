"""Summaries and figures of the reserve analyses (cross_cohort.py, other_tasks.py).

All statistics are descriptive. Intervals: percentile bootstrap 95% CIs with 2,000 resamples and seed 20261018,
resampling the independent units: contrasts for the cross-cohort pairs (pairs within a contrast share studies),
studies or outcome-study pairs for the real multiclass and regression data, datasets for the null subset.
Outputs: results/reserve_summary.json and figures/reserve_*.{png,pdf}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
SEED, B = 20261018, 2000
METHODS = ("top_k", "rfe", "spls", "pls_da", "opls_da")
NAME = {"top_k": "Top-q", "rfe": "RFE", "spls": "HT-sPLS", "pls_da": "PLS-DA", "opls_da": "OPLS-DA"}
REG_NAME = {"top_k": "Top-q", "rfe": "RFE (ridge)", "spls": "HT-sPLS", "pls_da": "PLS", "opls_da": "OPLS"}


def _boot(values: np.ndarray, clusters: np.ndarray | None, stat) -> list[float]:
    rng = np.random.default_rng(SEED)
    if clusters is None:
        draws = [stat(values[rng.integers(0, values.size, values.size)]) for _ in range(B)]
    else:
        keys = np.unique(clusters)
        members = [np.flatnonzero(clusters == k) for k in keys]
        draws = [stat(values[np.concatenate([members[i] for i in rng.integers(0, keys.size, keys.size)])])
                 for _ in range(B)]
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def summary(values, clusters=None, stat=np.median) -> dict:
    v = np.asarray(values, dtype=float)
    keep = np.isfinite(v)
    v, cl = v[keep], None if clusters is None else np.asarray(clusters)[keep]
    return {"n": int(v.size), "median": float(np.median(v)), "mean": float(np.mean(v)),
            "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)), "ci": _boot(v, cl, stat)}


def share(mask, clusters=None) -> dict:
    m = np.asarray(mask, dtype=float)
    return {"share": float(m.mean()), "count": int(m.sum()), "n": int(m.size), "ci": _boot(m, clusters, np.mean)}


# ------------------------------------------------------------------ A: cross-cohort

def cross_cohort_frame() -> pd.DataFrame:
    recs = json.loads((RES / "cross_cohort_records.json").read_text())
    bad = [r for r in recs if "error" in r]
    if bad:
        raise RuntimeError(f"failed pairs: {[(r['train'], r['test'], r['error']) for r in bad]}")
    rows = []
    for r in recs:
        for m, v in r["methods"].items():
            base = {k: r[k] for k in ("contrast", "train", "test", "n_train", "n_test", "genera_shared")}
            rows.append({**base, "method": m, "labels": "observed", **{k: v["observed"][k] for k in
                                                                      ("pre_split", "fold_contained", "external")}})
            rows += [{**base, "method": m, "labels": f"perm{p['labels']}",
                      **{k: p[k] for k in ("pre_split", "fold_contained", "external")}} for p in v["permuted"]]
    return pd.DataFrame(rows)


def cross_cohort(df: pd.DataFrame) -> dict:
    out = {"pairs": int(df[df.labels == "observed"].groupby("method").size().iloc[0]),
           "contrasts": int(df.contrast.nunique()), "study_contrasts": int(df[["contrast", "train"]].drop_duplicates()
                                                                            .shape[0]),
           "genera_shared": [int(df.genera_shared.min()), int(df.genera_shared.max())], "methods": {}}
    obs = df[(df.labels == "observed") & (df.method == METHODS[0])]
    out["contrast_studies"] = {c: sorted(set(g.train) | set(g.test)) for c, g in obs.groupby("contrast")}
    out["contrast_genera"] = {c: [int(g.genera_shared.min()), int(g.genera_shared.max())]
                              for c, g in obs.groupby("contrast")}
    out["n_train"] = [int(obs.n_train.min()), int(obs.n_train.max())]
    out["n_test"] = [int(obs.n_test.min()), int(obs.n_test.max())]
    for m in METHODS:
        d = df[(df.method == m) & (df.labels == "observed")]
        cl = d.contrast.to_numpy()
        e_pre, e_fc = (d.pre_split - d.external).to_numpy(), (d.fold_contained - d.external).to_numpy()
        closer = np.abs(e_fc) < np.abs(e_pre)
        nul = df[(df.method == m) & (df.labels != "observed")]
        per_pair = nul.groupby(["contrast", "train", "test"])[["pre_split", "fold_contained", "external"]].mean()
        ncl = per_pair.index.get_level_values(0).to_numpy()
        out["methods"][m] = {
            "pre_split": summary(d.pre_split, cl), "fold_contained": summary(d.fold_contained, cl),
            "external": summary(d.external, cl),
            "error_pre_split": summary(e_pre, cl), "error_fold_contained": summary(e_fc, cl),
            "over_010": {"pre_split": share(e_pre > 0.10, cl), "fold_contained": share(e_fc > 0.10, cl)},
            "over_020": {"pre_split": share(e_pre > 0.20, cl), "fold_contained": share(e_fc > 0.20, cl)},
            "fold_contained_closer": share(closer, cl),
            "rho_external": {"pre_split": float(spearmanr(d.pre_split, d.external).statistic),
                             "fold_contained": float(spearmanr(d.fold_contained, d.external).statistic)},
            "high_cv_low_external": {"pre_split": int(((d.pre_split >= 0.8) & (d.external < 0.7)).sum()),
                                     "fold_contained": int(((d.fold_contained >= 0.8) & (d.external < 0.7)).sum())},
            "null": {k: summary(per_pair[k], ncl) for k in ("pre_split", "fold_contained", "external")},
            "by_contrast": {c: {"pairs": int((cl == c).sum()), "external": float(np.median(d.external[cl == c])),
                                "error_pre_split": float(np.median(e_pre[cl == c])),
                                "error_fold_contained": float(np.median(e_fc[cl == c]))} for c in np.unique(cl)},
        }
    return out


def figure_cross_cohort(df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.6), sharex=True, sharey=True, constrained_layout=True)
    for ax, m in zip(axes, METHODS):
        d = df[(df.method == m) & (df.labels == "observed")]
        ax.plot([0.3, 1], [0.3, 1], color="0.6", lw=0.8, zorder=0)
        ax.axvline(0.5, color="0.85", lw=0.6, zorder=0)
        ax.scatter(d.external, d.pre_split, s=9, color="#B2182B", alpha=0.8, label="Pre-split CV", lw=0)
        ax.scatter(d.external, d.fold_contained, s=9, color="#2166AC", alpha=0.8, label="Fold-contained CV", lw=0)
        ax.set_title(NAME[m], fontsize=9)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.3, 1.02)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("External AUC (other study)", fontsize=8)
    axes[0].set_ylabel("Cross-validated AUC (training study)", fontsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=7, frameon=False)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


# ------------------------------------------------------------------ B and C: other tasks

def _task_frame(records: list[dict], key: str | None, id_fields: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for r in records:
        if "error" in r:
            raise RuntimeError(f"failed job: {r}")
        methods = r[key] if key else r["methods"]
        ident = {f: r.get(f) for f in id_fields}
        for m, v in methods.items():
            if "observed" in v:
                rows.append({**ident, "method": m, "labels": "observed",
                             "pre_split": v["observed"]["pre_split"], "fold_contained": v["observed"]["fold_contained"]})
            for p in v.get("permuted", v.get("null", [])):
                rows.append({**ident, "method": m, "labels": f"perm{p['labels']}", "pre_split": p["pre_split"],
                             "fold_contained": p["fold_contained"]})
    return pd.DataFrame(rows)


def task_summary(df: pd.DataFrame, unit: list[str], methods) -> dict:
    out = {}
    for m in methods:
        d = df[df.method == m]
        obs = d[d.labels == "observed"]
        nul = d[d.labels != "observed"].groupby(unit)[["pre_split", "fold_contained"]].mean()
        row = {"null": {k: summary(nul[k]) for k in ("pre_split", "fold_contained")},
               "null_gap": summary(nul.pre_split - nul.fold_contained)}
        if len(obs):
            row["observed"] = {k: summary(obs[k]) for k in ("pre_split", "fold_contained")}
            row["observed_gap"] = summary(obs.pre_split - obs.fold_contained)
            row["per_unit"] = obs[unit + ["pre_split", "fold_contained"]].to_dict(orient="records")
        out[m] = row
    return out


def figure_task(real: pd.DataFrame | None, null: pd.DataFrame | None, unit_real: list[str], methods, names: dict,
                metric: str, chance: float, path: Path, floor: float | None = None) -> None:
    """Left: real data, observed labels (one line per study). Middle: real data, mean over permutations. Right:
    null subset, per-dataset mean over null label vectors. With ``floor``, values below it are drawn at the floor
    as triangles, and the figure says how many."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_null = 0 if null is None else null.dataset_id.nunique()
    panels = [("Real data, observed", real, "observed", unit_real), ("Real data, permuted", real, "permuted", unit_real),
              (f"{n_null} datasets, null labels", null, "null", ["dataset_id"])]
    panels = [p for p in panels if p[1] is not None]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 2.8), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    clipped = 0
    for ax, (title, df, kind, unit) in zip(axes, panels):
        for k, m in enumerate(methods):
            d = df[df.method == m]
            if kind == "observed":
                d = d[d.labels == "observed"]
            else:
                d = d[d.labels != "observed"].groupby(unit)[["pre_split", "fold_contained"]].mean().reset_index()
            for x, col, values in ((3 * k, "#B2182B", d.pre_split.to_numpy()),
                                   (3 * k + 1, "#2166AC", d.fold_contained.to_numpy())):
                low = values < floor if floor is not None else np.zeros(values.size, bool)
                shown = np.where(low, floor, values) if floor is not None else values
                ax.scatter(np.full((~low).sum(), x), shown[~low], s=7, color=col, zorder=2, lw=0)
                ax.scatter(np.full(low.sum(), x), shown[low], s=14, color=col, marker="v", zorder=2, lw=0)
                clipped += int(low.sum())
            lines = zip(d.pre_split, d.fold_contained)
            for a, b in (((max(a, floor), max(b, floor)) for a, b in lines) if floor is not None else lines):
                ax.plot([3 * k, 3 * k + 1], [a, b], color="0.75", lw=0.5, zorder=1)
        ax.axhline(chance, color="0.4", lw=0.7, ls="--", zorder=0)
        if floor is not None:
            ax.set_ylim(bottom=floor - 0.05)
        ax.set_xticks([3 * k + 0.5 for k in range(len(methods))], [names[m] for m in methods], fontsize=7)
        ax.set_title(title, fontsize=8.5)
        ax.tick_params(axis="y", labelsize=7)
    axes[0].set_ylabel(metric, fontsize=8)
    axes[0].scatter([], [], s=7, color="#B2182B", label="Pre-split")
    axes[0].scatter([], [], s=7, color="#2166AC", label="Fold-contained")
    axes[0].legend(fontsize=6.5, frameon=False, loc="lower left")
    if clipped:
        lab = f"{floor:g}".replace("-", "−")
        axes[-1].text(0.99, 0.02, f"▼ {clipped} value{'s' if clipped > 1 else ''} below {lab}, drawn at {lab}",
                      transform=axes[-1].transAxes, ha="right", va="bottom", fontsize=6.5, color="0.3")
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    FIG.mkdir(exist_ok=True)
    result, frames = {}, {}
    if (RES / "cross_cohort_records.json").exists():
        cc = cross_cohort_frame()
        result["cross_cohort"] = cross_cohort(cc)
        # the cross-cohort analysis is reported in the article; its summary is also written on its own
        (RES / "cross_cohort_summary.json").write_text(json.dumps(result["cross_cohort"], indent=1))
        figure_cross_cohort(cc, FIG / "reserve_cross_cohort")
    for part, key, ident, unit, methods in (
            ("real_multiclass", None, ("study",), ["study"], ("top_k", "rfe", "pls_da")),
            ("real_regression", None, ("study", "outcome", "column"), ["study", "column"], METHODS)):
        path = RES / f"other_tasks_{part}.json"
        if path.exists():
            frames[part] = _task_frame(json.loads(path.read_text()), key, ident)
            result[part] = task_summary(frames[part], unit, methods)
    path = RES / "other_tasks_null_subset.json"
    if path.exists():
        recs = json.loads(path.read_text())
        # A job may fail only because its participants cannot support a participant-aware split (E-MTAB-5606 has
        # 8 participants, so balanced three-class labels leave 2 in one class); such jobs are excluded and listed.
        failed = [r for r in recs if "error" in r]
        if any("insufficient groups" not in r["error"] for r in failed):
            raise RuntimeError(f"null subset jobs failed: {failed}")
        result["null_excluded"] = failed
        recs = [r for r in recs if "error" not in r]
        frames["null_multiclass"] = _task_frame([r for r in recs if r.get("task") == "multiclass"], None,
                                                ("dataset_id",))
        frames["null_regression"] = _task_frame([r for r in recs if r.get("task") == "regression"], None,
                                                ("dataset_id",))
        if len(recs) + len(failed) != 200 or len(recs) != len(frames["null_multiclass"].dataset_id.unique()) + len(
                frames["null_regression"].dataset_id.unique()):
            raise RuntimeError("null subset records incomplete")
        result["null_multiclass"] = task_summary(frames["null_multiclass"], ["dataset_id"], ("top_k", "rfe", "pls_da"))
        result["null_regression"] = task_summary(frames["null_regression"], ["dataset_id"], METHODS)
    if "real_multiclass" in frames or "null_multiclass" in frames:
        figure_task(frames.get("real_multiclass"), frames.get("null_multiclass"), ["study"], ("top_k", "rfe", "pls_da"), NAME,
                    "Macro one-vs-rest AUC (pooled out-of-fold)", 0.5, FIG / "reserve_multiclass")
    if "real_regression" in frames or "null_regression" in frames:
        figure_task(frames.get("real_regression"), frames.get("null_regression"), ["study", "column"],
                    METHODS, REG_NAME, "Out-of-fold R²", 0.0, FIG / "reserve_regression", floor=-1.0)
    (RES / "reserve_summary.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
