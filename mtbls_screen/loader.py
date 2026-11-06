"""End-to-end loading of a MetaboLights study into (X, y).

The catalogue stage screens on a 64 KB Range request and is deliberately
permissive, so a catalogued study may still fail here: the assignment
file may not join to the sample file, the intensities may be absent, or
too few samples may survive the intersection. Every failure returns a
reason rather than an exception, because the attrition rate between
"catalogued" and "loadable" is itself a reportable property of the
repository.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fetch import BASE, get_binary
from isatab import (
    MIN_FEATURES, MIN_PER_CLASS, MIN_SAMPLES, binary_contrasts, log_scale,
    maf_matrix, norm_name, read_tsv,
)

ALIAS_COLUMNS = (
    "Sample Name", "MS Assay Name", "NMR Assay Name",
    "Raw Spectral Data File", "Derived Spectral Data File",
    "Free Induction Decay Data File",
)


@dataclass(frozen=True)
class Dataset:
    study: str
    X: np.ndarray                 # samples x features, log scale
    y: np.ndarray                 # binary
    samples: list[str]
    phenotype: str
    levels: tuple[str, str]
    maf: str


def fetch_table(study: str, fname: str):
    p = get_binary(f"{BASE}/{study}/{fname}", key=f"f_{study}_{fname}")
    return read_tsv(p) if p else None


def pick_contrast(contrasts):
    """The most balanced binary factor; alphabetical tie-break.

    Fixed in advance so that one contrast is analysed per study.
    Choosing the strongest would turn the screen into a search over
    contrasts and inflate every downstream proportion.
    """
    if not contrasts:
        return None
    return min(contrasts,
               key=lambda c: (abs(c.n_pos - c.n_neg) / (c.n_pos + c.n_neg),
                              c.name))


def build_alias(assay: pd.DataFrame) -> dict[str, str]:
    """Map every name a sample is known by back to its sample-file name."""
    alias: dict[str, str] = {}
    for col in ALIAS_COLUMNS:
        if col not in assay.columns:
            continue
        for src, dst in zip(assay[col], assay["Sample Name"]):
            if pd.notna(src):
                for part in str(src).split(";"):
                    alias.setdefault(norm_name(part.split("/")[-1]),
                                     str(dst).strip())
    alias.pop("", None)
    return alias


def load_study(entry: dict) -> tuple[Dataset | None, str]:
    """Return (dataset, reason). dataset is None unless reason == 'ok'."""
    study = entry["study"]

    samples = fetch_table(study, entry["sample_file"])
    if samples is None or "Sample Name" not in samples.columns:
        return None, "no_sample_file"
    samples["Sample Name"] = samples["Sample Name"].astype(str).str.strip()

    ct = pick_contrast(binary_contrasts(samples))
    if ct is None:
        return None, "no_binary_phenotype"

    assay = None
    for a in entry["assay_files"]:
        t = fetch_table(study, a)
        if t is not None and "Sample Name" in t.columns:
            t["Sample Name"] = t["Sample Name"].astype(str).str.strip()
            assay = t
            break
    if assay is None:
        return None, "no_assay_file"
    alias = build_alias(assay)

    mafs = list(entry["maf_files"])
    if "Metabolite Assignment File" in assay.columns:
        named = [m for m in assay["Metabolite Assignment File"].dropna()
                 .astype(str).unique() if m in set(mafs)]
        mafs = named + [m for m in mafs if m not in named]

    X = maf_name = None
    for m in mafs:
        maf = fetch_table(study, m)
        if maf is None:
            continue
        X = maf_matrix(maf, alias)
        if X is not None:
            maf_name = m
            break
    if X is None:
        return None, "no_usable_matrix"

    lab = samples.set_index("Sample Name")[ct.name].astype(str).str.strip()
    lab = lab[lab.isin([ct.level_pos, ct.level_neg])]
    lab = lab[~lab.index.duplicated(keep="first")]
    idx = [s for s in X.index if s in lab.index]
    if len(idx) < MIN_SAMPLES:
        return None, "too_few_aligned"

    y = (lab.loc[idx] == ct.level_pos).astype(int).to_numpy()
    if min(np.bincount(y, minlength=2)) < MIN_PER_CLASS:
        return None, "class_too_small"

    V = log_scale(X.loc[idx])
    keep = np.isfinite(V).all(axis=0) & (V.std(axis=0) > 1e-12)
    V = V[:, keep]
    if V.shape[1] < MIN_FEATURES:
        return None, "too_few_features"

    return Dataset(study=study, X=V, y=y, samples=list(idx),
                   phenotype=ct.name,
                   levels=(ct.level_pos, ct.level_neg),
                   maf=maf_name or ""), "ok"
