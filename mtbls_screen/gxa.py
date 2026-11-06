"""EBI Expression Atlas as a second and third omics platform.

Expression Atlas curates experiments into a uniform structure: an
expression matrix per experiment and a `condensed-sdrf` table that
records, per assay, both characteristics and the declared experimental
factors. That makes the phenotype contrast machine-readable in a way
raw GEO is not, and it covers transcriptomics and proteomics with the
same interface.

Files on the FTP mirror, per accession:
    {ACC}-raw-counts.tsv                    RNA-seq
    {ACC}_{ARRAY}-normalized-expressions.tsv  microarray
    {ACC}.condensed-sdrf.tsv                assay -> factor/characteristic
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fetch import get, get_binary

GXA_JSON = "https://www.ebi.ac.uk/gxa/json/experiments"
GXA_FTP = ("https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/"
           "experiments")

MIN_SAMPLES = 30
MIN_PER_CLASS = 10
MIN_FEATURES = 20
# Transcriptomes carry tens of thousands of genes. Keeping the most
# variable subset is unsupervised, so it cannot leak the labels; the
# PCA arm of this study measures that class of optimism directly and
# finds it negligible. The cap exists purely to bound compute.
MAX_FEATURES = 5000

# Factor names that describe a biological grouping rather than a time,
# a site or a technical property. Mirrors the metabolomics rule so the
# platforms are treated identically.
FACTOR_ELIGIBLE = re.compile(
    r"disease|diagnos|patholog|tumou?r|cancer|carcinom|"
    r"compound|treatment|dose|drug|irradiat|stimul|"
    r"genotype|strain|variant|knock|transgen|mutant|"
    r"\bsex\b|gender|phenotype|infect|immun|"
    r"clinical information|disease staging",
    re.I,
)
FACTOR_INELIGIBLE = re.compile(
    r"time|age|developmental stage|organism part|cell line|cell type|"
    r"sampling|block|batch|individual|replicate|growth condition|"
    r"organism$|biosource|media|culture",
    re.I,
)


def eligible_factor(name: str) -> bool:
    if FACTOR_INELIGIBLE.search(name):
        return False
    return bool(FACTOR_ELIGIBLE.search(name))


@dataclass(frozen=True)
class GxaDataset:
    accession: str
    X: np.ndarray
    y: np.ndarray
    factor: str
    levels: tuple[str, str]
    platform: str
    species: str


def list_files(acc: str) -> list[str]:
    html = get(f"{GXA_FTP}/{acc}/", key=f"gxa_dir_{acc}.html")
    if not html:
        return []
    return [h for h in re.findall(r'href="([^"?/][^"]*)"', html)
            if not h.startswith(("/", "http"))]


def read_sdrf(acc: str) -> pd.DataFrame | None:
    """Per-assay factor and characteristic values.

    The condensed SDRF is ragged (an optional ontology column), so the
    width is fixed explicitly rather than inferred.
    """
    p = get_binary(f"{GXA_FTP}/{acc}/{acc}.condensed-sdrf.tsv",
                   key=f"gxa_sdrf_{acc}.tsv")
    if not p:
        return None
    try:
        d = pd.read_csv(p, sep="\t", header=None, names=range(8),
                        engine="python", dtype=str, on_bad_lines="skip")
    except Exception:
        return None
    d = d.rename(columns={2: "assay", 3: "kind", 4: "name", 5: "value"})
    return d.dropna(subset=["assay", "kind", "name"])


def contrasts_from_sdrf(sdrf: pd.DataFrame) -> list[tuple[str, str, str,
                                                          int, int]]:
    """Binary contrasts available from the declared factors."""
    fac = sdrf[sdrf["kind"].str.lower().eq("factor")]
    out = []
    for name, g in fac.groupby("name"):
        if not eligible_factor(str(name)):
            continue
        vc = g.drop_duplicates("assay")["value"].dropna().value_counts()
        if len(vc) < 2:
            continue
        (l1, n1), (l2, n2) = list(vc.items())[:2]
        if min(n1, n2) < MIN_PER_CLASS:
            continue
        out.append((str(name), str(l1), str(l2), int(n1), int(n2)))
    return out


def read_matrix(acc: str, files: list[str]) -> pd.DataFrame | None:
    """Expression matrix as samples x features."""
    cand = ([f for f in files if f.endswith("-raw-counts.tsv")]
            + [f for f in files if f.endswith("-normalized-expressions.tsv")]
            + [f for f in files if f.endswith("-tpms.tsv")]
            + [f for f in files if f.endswith("-fpkms.tsv")])
    for f in cand:
        p = get_binary(f"{GXA_FTP}/{acc}/{f}", key=f"gxa_mat_{acc}_{f}")
        if not p:
            continue
        try:
            d = pd.read_csv(p, sep="\t", low_memory=False)
        except Exception:
            continue
        meta = [c for c in d.columns
                if re.search(r"gene|design element|probe|protein|id$|name$",
                             str(c), re.I)]
        sample_cols = [c for c in d.columns if c not in meta]
        if len(sample_cols) < MIN_SAMPLES:
            continue
        M = d[sample_cols].apply(pd.to_numeric, errors="coerce").T
        M = M.loc[:, M.notna().mean(axis=0) >= 0.5]
        if M.shape[1] < MIN_FEATURES:
            continue
        return M.fillna(M.median(axis=0))
    return None


def normalise(M: pd.DataFrame, is_counts: bool) -> np.ndarray:
    """Counts per million then log, or log for already-scaled values."""
    V = M.to_numpy(dtype=float)
    V = np.where(np.isfinite(V), V, np.nan)
    if is_counts:
        tot = np.nansum(V, axis=1, keepdims=True)
        tot[tot <= 0] = 1.0
        V = V / tot * 1e6
    pos = V[np.isfinite(V) & (V > 0)]
    floor = pos.min() if pos.size else 1.0
    V = np.where(np.isfinite(V), np.maximum(V, floor), floor)
    return np.log2(V) if V.max() > 50 else V


def cap_features(V: np.ndarray) -> np.ndarray:
    keep = np.isfinite(V).all(axis=0) & (V.std(axis=0) > 1e-12)
    V = V[:, keep]
    if V.shape[1] > MAX_FEATURES:
        V = V[:, np.argsort(-V.std(axis=0))[:MAX_FEATURES]]
    return V


def load_experiment(entry: dict) -> tuple[GxaDataset | None, str]:
    acc = entry["accession"]
    files = list_files(acc)
    if not files:
        return None, "no_ftp_listing"

    sdrf = read_sdrf(acc)
    if sdrf is None:
        return None, "no_sdrf"
    cands = contrasts_from_sdrf(sdrf)
    if not cands:
        return None, "no_eligible_factor"
    name, l1, l2, n1, n2 = min(
        cands, key=lambda c: (abs(c[3] - c[4]) / (c[3] + c[4]), c[0]))

    M = read_matrix(acc, files)
    if M is None:
        return None, "no_matrix"

    fac = sdrf[(sdrf["kind"].str.lower() == "factor")
               & (sdrf["name"] == name)].drop_duplicates("assay")
    lab = fac.set_index("assay")["value"]
    lab = lab[lab.isin([l1, l2])]
    idx = [s for s in M.index if s in lab.index]
    if len(idx) < MIN_SAMPLES:
        return None, "too_few_aligned"

    y = (lab.loc[idx] == l1).astype(int).to_numpy()
    if min(np.bincount(y, minlength=2)) < MIN_PER_CLASS:
        return None, "class_too_small"

    is_counts = any(f.endswith("-raw-counts.tsv") for f in files)
    V = cap_features(normalise(M.loc[idx], is_counts))
    if V.shape[1] < MIN_FEATURES:
        return None, "too_few_features"

    return GxaDataset(accession=acc, X=V, y=y, factor=name,
                      levels=(l1, l2),
                      platform=entry.get("type", ""),
                      species=entry.get("species", "")), "ok"
