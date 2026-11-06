"""Exploratory design-only model of the pre-split null, added before submission to Patterns.

Question: given only the number of samples (n) and of features (p, counted before preprocessing), how high
does a pre-split pipeline push the AUC on pure noise? The specification below was fixed before any model was
fitted; the analysis itself was chosen after the formal results were known and is exploratory (Table S12).

- Data: the 1,030 exchangeable datasets. For each method, the dataset's mean pre-split AUC over its 199
  formal permutations (``results/nulls_by_design.csv``).
- Outcome: Somers' D = 2 AUC - 1 (0, chance; 1, perfect separation), clipped to [0.001, 0.999].
- Model, per method: fractional logit (quasi-binomial GLM) with natural cubic regression splines of 4 degrees of
  freedom in log10 n and in log10 p, additive; boundary knots fixed at n = 30 and 2,000 and p = 20 and 200,000.
- Comparators: the training mean (no design information) and a fractional logit linear in log10 n and log10 p.
- Evaluation: leave-one-repository-out (4 folds) and 10-fold cross-validation (seed 20261019). Spline knots are
  placed within each training fold. MAE, RMSE and out-of-sample R^2 (against the training mean) on the AUC
  scale; the 10th and 90th percentiles of the 10-fold out-of-fold errors give an empirical 80% range.
- Final model: fitted to all 1,030 datasets and predicted on a 60 x 60 grid in log10 n and log10 p. Grid points
  with fewer than 5 datasets within 0.3 log10 units are marked unsupported and are not shown or used.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
TOOL_DATA = HERE.parent / "tool" / "src" / "presplit_audit" / "data"
METHODS = ["top_k", "rfe", "spls", "pls_da", "opls_da"]
# Boundary knots fixed a priori (inclusion criteria n >= 30 and p >= 20; upper limits above any dataset), so that
# no fold's knots depend on held-out designs; interior knots are training-fold quantiles.
N_BOUNDS, P_BOUNDS = (np.log10(30), np.log10(2_000)), (np.log10(20), np.log10(200_000))
SPLINE = (f"D ~ cr(lgn, df=4, constraints='center', lower_bound={N_BOUNDS[0]}, upper_bound={N_BOUNDS[1]})"
          f" + cr(lgp, df=4, constraints='center', lower_bound={P_BOUNDS[0]}, upper_bound={P_BOUNDS[1]})")
LINEAR = "D ~ lgn + lgp"
SEED, K, EPS = 20261019, 10, 1e-3
GRID_N = np.linspace(np.log10(30), np.log10(1200), 60)
GRID_P = np.linspace(np.log10(20), np.log10(180_000), 60)
RADIUS, MIN_NEAR = 0.3, 5
REPRESENTATIVE_N, REPRESENTATIVE_P = [30, 50, 100, 200, 500], [100, 1_000, 10_000, 50_000]


def load() -> pd.DataFrame:
    d = pd.read_csv(OUT / "nulls_by_design.csv")
    d["lgn"], d["lgp"] = np.log10(d.n_samples), np.log10(d.n_features)
    d["D"] = np.clip(2 * d.null_pre - 1, EPS, 1 - EPS)
    return d


def fit(train: pd.DataFrame, formula: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                          # quasi-binomial with fractional outcomes
        return smf.glm(formula, data=train, family=sm.families.Binomial()).fit()


def to_auc(model, frame: pd.DataFrame) -> np.ndarray:
    return (1 + np.asarray(model.predict(frame), dtype=float)) / 2


def metrics(obs: np.ndarray, pred: np.ndarray, base: np.ndarray) -> dict:
    err = obs - pred
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2_oos": float(1 - np.sum(err ** 2) / np.sum((obs - base) ** 2)),
            "spearman": float(spearmanr(obs, pred).statistic)}


def cross_validate(d: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[dict, pd.DataFrame]:
    pred = {k: np.full(len(d), np.nan) for k in ("spline", "linear", "mean")}
    for tr, te in folds:
        train, test = d.iloc[tr], d.iloc[te]
        pred["spline"][te] = to_auc(fit(train, SPLINE), test)
        pred["linear"][te] = to_auc(fit(train, LINEAR), test)
        pred["mean"][te] = train.null_pre.mean()
    obs = d.null_pre.to_numpy()
    assert all(np.isfinite(v).all() for v in pred.values())
    out = {k: metrics(obs, pred[k], pred["mean"]) for k in ("spline", "linear")}
    out["mean"] = {"mae": float(np.mean(np.abs(obs - pred["mean"]))),
                   "rmse": float(np.sqrt(np.mean((obs - pred["mean"]) ** 2)))}
    return out, pd.DataFrame({"dataset_id": d.dataset_id, "repository": d.repository, "observed": obs,
                              **{f"pred_{k}": v for k, v in pred.items()}})


def supported(d: pd.DataFrame, lgn: np.ndarray, lgp: np.ndarray) -> np.ndarray:
    pts = d.drop_duplicates("dataset_id")[["lgn", "lgp"]].to_numpy()
    dist = np.sqrt((lgn[..., None] - pts[:, 0]) ** 2 + (lgp[..., None] - pts[:, 1]) ** 2)
    return (dist <= RADIUS).sum(axis=-1) >= MIN_NEAR


def main() -> None:
    data = load()
    rng_folds = KFold(n_splits=K, shuffle=True, random_state=SEED)
    result = {"specification": __doc__.strip(), "n_datasets": int(data.dataset_id.nunique()), "methods": {}}
    grids, cv_rows = [], []
    gn, gp = np.meshgrid(GRID_N, GRID_P, indexing="ij")
    support = supported(data, gn, gp)
    for m in METHODS:
        d = data[data.method == m].reset_index(drop=True)
        loro = [(np.flatnonzero(d.repository != r), np.flatnonzero(d.repository == r))
                for r in sorted(d.repository.unique())]
        kfold = list(rng_folds.split(d))
        res_loro, pred_loro = cross_validate(d, loro)
        res_k, pred_k = cross_validate(d, kfold)
        err = pred_k.observed - pred_k.pred_spline
        final = fit(d, SPLINE)
        grid = pd.DataFrame({"lgn": gn.ravel(), "lgp": gp.ravel()})
        auc = to_auc(final, grid).reshape(gn.shape)
        # shape inside the supported region, as net changes (not per grid step): the largest rise along n
        # from any design to a larger n, the largest fall along p after its running maximum, and where the
        # prediction peaks along p
        rise_n = max((float(np.max(v - np.minimum.accumulate(v))) for v in
                      (auc[support[:, j], j] for j in range(auc.shape[1])) if v.size > 1), default=0.0)
        rows_p = [(auc[i, support[i]], GRID_P[support[i]]) for i in range(auc.shape[0]) if support[i].sum() > 1]
        fall_p = max(float(np.max(np.maximum.accumulate(v) - v)) for v, _ in rows_p)
        peak_p = float(np.median([10 ** lp[np.argmax(v)] for v, lp in rows_p]))
        rep = []
        for nn in REPRESENTATIVE_N:
            for pp in REPRESENTATIVE_P:
                point = pd.DataFrame({"lgn": [np.log10(nn)], "lgp": [np.log10(pp)]})
                ok = bool(supported(data, np.log10(np.array([nn])), np.log10(np.array([pp])))[0])
                rep.append({"n": nn, "p": pp, "auc": float(to_auc(final, point)[0]), "supported": ok})
        result["methods"][m] = {
            "loro": res_loro, "kfold": res_k,
            "loro_by_repository": {r: {"n": int((pred_loro.repository == r).sum()),
                                       "mae_spline": float(np.mean(np.abs(g.observed - g.pred_spline))),
                                       "mae_mean": float(np.mean(np.abs(g.observed - g.pred_mean)))}
                                   for r, g in pred_loro.groupby("repository")},
            "error_q10_q90": [float(np.quantile(err, 0.1)), float(np.quantile(err, 0.9))],
            "max_net_rise_along_n": rise_n, "max_net_fall_along_p": fall_p, "median_p_at_peak": peak_p,
            "mean_observed": float(d.null_pre.mean()),
            "mean_predicted_10_fold": float(pred_k.pred_spline.mean()),
            "representative": rep,
            "deviance_explained": float(1 - final.deviance / final.null_deviance),
        }
        grids.append(pd.DataFrame({"method": m, "lgn": gn.ravel(), "lgp": gp.ravel(), "auc": auc.ravel(),
                                   "supported": support.ravel()}))
        cv_rows += [pred_loro.assign(method=m, scheme="leave_one_repository_out"),
                    pred_k.assign(method=m, scheme="10_fold")]
    OUT.mkdir(exist_ok=True)
    (OUT / "risk_model.json").write_text(json.dumps(result, indent=1))
    pd.concat(grids).to_csv(OUT / "risk_grid.csv", index=False)
    pd.concat(cv_rows).to_csv(OUT / "risk_cv_predictions.csv", index=False)
    # compact grid shipped with presplit-audit (interpolated by presplit_audit.risk)
    TOOL_DATA.mkdir(exist_ok=True)
    g = pd.concat(grids)
    payload = {"source": "Exploratory design-only model of the pre-split null fitted to 1,030 public omics "
                         "datasets (Patterns article, Table S17); see analyses/risk_model.py.",
               "log10_n": [round(v, 6) for v in GRID_N], "log10_p": [round(v, 6) for v in GRID_P],
               "supported": support.astype(int).tolist(),
               "methods": {m: {"auc": np.round(g[g.method == m].auc.to_numpy().reshape(gn.shape), 4).tolist(),
                               "error_q10_q90": result["methods"][m]["error_q10_q90"],
                               "mae_10_fold": result["methods"][m]["kfold"]["spline"]["mae"]}
                           for m in METHODS}}
    (TOOL_DATA / "risk_grid.json").write_text(json.dumps(payload, separators=(",", ":")))
    for m in METHODS:
        r = result["methods"][m]
        print(f"{m:8s} LORO MAE spline {r['loro']['spline']['mae']:.3f} linear {r['loro']['linear']['mae']:.3f} "
              f"mean {r['loro']['mean']['mae']:.3f} R2 {r['loro']['spline']['r2_oos']:.2f} | 10-fold MAE "
              f"{r['kfold']['spline']['mae']:.3f} (mean {r['kfold']['mean']['mae']:.3f}) R2 "
              f"{r['kfold']['spline']['r2_oos']:.2f} | 80% err {r['error_q10_q90'][0]:+.3f}/{r['error_q10_q90'][1]:+.3f}"
              f" | rise along n {r['max_net_rise_along_n']:.3f}, fall along p {r['max_net_fall_along_p']:.3f}, "
              f"peak p {r['median_p_at_peak']:,.0f}")


if __name__ == "__main__":
    main()
