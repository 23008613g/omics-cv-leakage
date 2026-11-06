"""Metabolomics Workbench as an independent replication repository.

Three of the four platforms in this study are hosted by EBI, which
invites the objection that the results reflect one institution's
curation rather than a property of the analyses. Metabolomics Workbench
(NIH / UC San Diego) holds the same *kind* of data as MetaboLights under
entirely separate submission standards and curation, so running the
identical grid there tests that objection directly rather than adding a
fifth assay type.

REST endpoints, per study:
    /summary   title, species, institute, analysis type, sample count
    /factors   per-sample factor string, "Name:Value | Name:Value"
    /data      per-metabolite record with DATA = {sample_id: value}
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fetch import get
from isatab import PHENOTYPE_ELIGIBLE, PHENOTYPE_INELIGIBLE

REST = "https://www.metabolomicsworkbench.org/rest/study/study_id"

MIN_SAMPLES = 30
MIN_PER_CLASS = 10
MIN_FEATURES = 20


@dataclass(frozen=True)
class MwDataset:
    study: str
    X: np.ndarray
    y: np.ndarray
    factor: str
    levels: tuple[str, str]
    institute: str
    analysis: str


def _json(url: str, key: str):
    txt = get(url, key=key)
    if not txt:
        return None
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return None


def list_studies(max_prefix: int = 40) -> list[dict]:
    """Enumerate studies by ST%04d prefix.

    There is no bulk listing endpoint that returns reliably, but a
    truncated study id acts as a prefix query, so the catalogue is
    assembled a hundred studies at a time.
    """
    out: dict[str, dict] = {}
    for i in range(max_prefix):
        d = _json(f"{REST}/ST{i:04d}/summary", key=f"mw_sum_{i:04d}.json")
        if not isinstance(d, dict):
            continue
        rows = d.values() if "study_id" not in d else [d]
        for r in rows:
            if isinstance(r, dict) and r.get("study_id"):
                out[r["study_id"]] = r
    return list(out.values())


def eligible_factor(name: str) -> bool:
    if PHENOTYPE_INELIGIBLE.search(name):
        return False
    return bool(PHENOTYPE_ELIGIBLE.search(name))


def parse_factors(study: str) -> pd.DataFrame | None:
    """Per-sample factor table from the pipe-delimited factor string."""
    d = _json(f"{REST}/{study}/factors", key=f"mw_fac_{study}.json")
    if not isinstance(d, dict) or not d:
        return None
    rows = {}
    for r in d.values():
        if not isinstance(r, dict):
            continue
        sid = r.get("local_sample_id")
        if not sid:
            continue
        vals = {}
        for part in str(r.get("factors", "")).split("|"):
            if ":" not in part:
                continue
            k, v = part.split(":", 1)
            vals[k.strip()] = v.strip()
        if vals:
            rows[str(sid).strip()] = vals
    return pd.DataFrame.from_dict(rows, orient="index") if rows else None


def parse_data(study: str) -> tuple[pd.DataFrame | None, str]:
    """Samples x metabolites for the analysis with the most metabolites.

    A study may hold several analyses (ionisation modes, platforms).
    Mixing them would concatenate incomparable measurement scales, so
    the largest single analysis is used.
    """
    d = _json(f"{REST}/{study}/data", key=f"mw_dat_{study}.json")
    if not isinstance(d, dict) or not d:
        return None, ""

    by_analysis: dict[str, dict[str, dict]] = {}
    for r in d.values():
        if not isinstance(r, dict) or not isinstance(r.get("DATA"), dict):
            continue
        an = str(r.get("analysis_id", "AN"))
        name = str(r.get("metabolite_name") or r.get("metabolite_id"))
        by_analysis.setdefault(an, {})[name] = r["DATA"]
    if not by_analysis:
        return None, ""

    an = max(by_analysis, key=lambda a: len(by_analysis[a]))
    M = pd.DataFrame(by_analysis[an]).apply(pd.to_numeric, errors="coerce")
    M.index = [str(i).strip() for i in M.index]
    return M, an


def load_study(entry: dict) -> tuple[MwDataset | None, str]:
    study = entry["study_id"]

    fac = parse_factors(study)
    if fac is None or fac.empty:
        return None, "no_factors"

    M, an = parse_data(study)
    if M is None:
        return None, "no_matrix"

    # The factor table often covers more samples than the chosen
    # analysis measured, so the contrast is selected on the intersection.
    # Choosing it first would pick levels that mostly sit outside the
    # matrix and then fail alignment for a purely bookkeeping reason.
    shared = [s for s in M.index if s in fac.index]
    if len(shared) < MIN_SAMPLES:
        return None, "too_few_aligned"
    fac = fac.loc[shared]

    cands = []
    for col in fac.columns:
        if not eligible_factor(str(col)):
            continue
        vc = fac[col].dropna().astype(str).str.strip().value_counts()
        if len(vc) < 2:
            continue
        (l1, n1), (l2, n2) = list(vc.items())[:2]
        if min(n1, n2) < MIN_PER_CLASS or (n1 + n2) < MIN_SAMPLES:
            continue
        cands.append((str(col), str(l1), str(l2), int(n1), int(n2)))
    if not cands:
        return None, "no_eligible_factor"
    name, l1, l2, _, _ = min(
        cands, key=lambda c: (abs(c[3] - c[4]) / (c[3] + c[4]), c[0]))

    lab = fac[name].astype(str).str.strip()
    lab = lab[lab.isin([l1, l2])]
    idx = [s for s in shared if s in lab.index]
    if len(idx) < MIN_SAMPLES:
        return None, "too_few_aligned"

    y = (lab.loc[idx] == l1).astype(int).to_numpy()
    if min(np.bincount(y, minlength=2)) < MIN_PER_CLASS:
        return None, "class_too_small"

    V = M.loc[idx].to_numpy(dtype=float)
    keep_col = np.isfinite(V).mean(axis=0) >= 0.5
    V = V[:, keep_col]
    if V.shape[1] < MIN_FEATURES:
        return None, "too_few_features"
    med = np.nanmedian(V, axis=0)
    V = np.where(np.isfinite(V), V, med)

    pos = V[V > 0]
    floor = pos.min() if pos.size else 1.0
    V = np.log2(np.maximum(V, floor))

    keep_col = V.std(axis=0) > 1e-12
    V = V[:, keep_col]
    if V.shape[1] < MIN_FEATURES:
        return None, "too_few_features"

    return MwDataset(study=study, X=V, y=y, factor=name, levels=(l1, l2),
                     institute=str(entry.get("institute", "")),
                     analysis=an), "ok"
