"""Parsing of ISA-Tab study/assay files and metabolite assignment files.

ISA-Tab is only loosely standardised in practice: column names, the key
used to join an assay to its data matrix, and the way factors are
encoded all vary between submissions. Everything here therefore fails
soft -- a study we cannot parse confidently is skipped and counted,
never guessed at, because a mis-parsed study would enter the screen as a
false positive.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_SAMPLES = 30
MIN_PER_CLASS = 10
# Targeted panels routinely quantify only a few dozen metabolites; the
# rank statistics used here are meaningful well below the size of an
# untargeted feature table.
MIN_FEATURES = 20

# Characteristics columns that describe the assay or the ontology rather
# than the participant, and so cannot serve as a phenotype.
NOT_PHENOTYPE = re.compile(
    r"^(organism|organism part|term source ref|term accession|unit|"
    r"variant|material type|sample type|technology|instrument|protocol|"
    r"derived spectral|raw spectral|assay|sample name|source name|"
    r"characteristics\[organism)",
    re.I,
)

# MAF columns that describe the feature, not a sample
MAF_META = re.compile(
    r"^(database_identifier|chemical_formula|smiles|inchi|metabolite_"
    r"identification|mass_to_charge|fragmentation|modifications|charge|"
    r"retention_time|taxid|species|database|database_version|reliability|"
    r"uri|search_engine|search_engine_score|smallmolecule_abundance|"
    r"chemical_shift|multiplicity|sample_measurement|.*_id|unnamed.*)",
    re.I,
)


def read_tsv(path) -> pd.DataFrame:
    """Read an ISA-Tab table.

    A handful of submissions are Latin-1 rather than UTF-8 -- often a
    stray degree sign or a non-breaking space in a protocol
    description. Failing on those would discard the whole study for a
    reason that has nothing to do with its data, so the encoding falls
    back rather than raising.
    """
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, sep="\t", dtype=str, low_memory=False,
                               on_bad_lines="skip", encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, sep="\t", dtype=str, low_memory=False,
                       on_bad_lines="skip", encoding="utf-8",
                       encoding_errors="replace")


def bracketed(col: str) -> str:
    m = re.search(r"\[(.*)\]", col)
    return m.group(1) if m else col


# ------------------------------------------------------------------
def numeric_order(series: pd.Series) -> np.ndarray | None:
    """Validate a candidate order column numerically.

    A genuine acquisition order is numeric and nearly all-distinct. Many
    columns whose *name* matches (e.g. "Run Number" holding a constant
    protocol id) fail this and must not be used.
    """
    v = pd.to_numeric(series, errors="coerce")
    if v.notna().sum() < MIN_SAMPLES:
        return None
    v = v.to_numpy(dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < MIN_SAMPLES:
        return None
    uniq = len(np.unique(v[ok])) / ok.sum()
    if uniq < 0.80:
        return None
    return v


MIN_DATE_LEVELS = 10


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse a timestamp column, retrying day-first for ambiguous forms
    such as 21/06/12, which are common in European submissions."""
    dt = pd.to_datetime(series, errors="coerce", format="mixed")
    if dt.notna().mean() < 0.5:
        alt = pd.to_datetime(series, errors="coerce", format="mixed",
                             dayfirst=True)
        if alt.notna().mean() > dt.notna().mean():
            return alt
    return dt


def datetime_order(series: pd.Series) -> np.ndarray | None:
    """Recover acquisition order from timestamps.

    Unlike an injection index, a timestamp column is expected to contain
    ties -- many studies record only the analysis date, so a whole day's
    runs share a value. Order is therefore coarse here, which is still
    informative about drift, so uniqueness is not required; only that
    the column resolves enough distinct time points to order samples at
    all.
    """
    dt = parse_dates(series)
    if dt.notna().sum() < MIN_SAMPLES:
        return None
    if dt.nunique() < MIN_DATE_LEVELS:
        return None
    v = dt.astype("int64").astype(float)
    v[dt.isna()] = np.nan
    return v.to_numpy()


# The claim under test concerns *biological* contrasts. A study whose
# grouping variable is itself a time, a site or a collection season is
# expected to align with acquisition date, and finding that it does is
# not evidence of a methodological error -- so those are not eligible
# as the phenotype.
PHENOTYPE_ELIGIBLE = re.compile(
    r"disease|diagnos|patholog|tumou?r|cancer|carcinom|malign|"
    r"treatment|therap|\bdrug\b|\bdose\b|placebo|\barm\b|intervention|"
    r"genotype|strain|mutant|knock|transgen|wild.?type|"
    r"\bsex\b|gender|"
    r"case|control|patient|healthy|"
    r"infect|challenge|exposure|responder|response|resistan|"
    r"phenotype|condition|diet|obes|diabet|smok",
    re.I,
)
PHENOTYPE_INELIGIBLE = re.compile(
    r"time|day|week|month|year|season|date|hour|"
    r"site|location|geograph|region|country|"
    r"batch|plate|well|order|instrument|column|extraction|vial|"
    r"harvest|collect|sampling|storage|"
    r"replicate|technical",
    re.I,
)


def eligible_phenotype(col: str) -> bool:
    name = bracketed(col)
    if PHENOTYPE_INELIGIBLE.search(name):
        return False
    return bool(PHENOTYPE_ELIGIBLE.search(name))


@dataclass(frozen=True)
class Contrast:
    """One binary phenotype contrast drawn from a study's factors."""
    name: str
    level_pos: str
    level_neg: str
    n_pos: int
    n_neg: int


def binary_contrasts(samples: pd.DataFrame) -> list[Contrast]:
    """Every Factor Value column that yields a usable binary contrast.

    Factors with more than two levels contribute their two most frequent
    levels; the remaining samples are dropped for that contrast.
    """
    out: list[Contrast] = []
    for col in samples.columns:
        # Studies are inconsistent about whether the grouping variable is
        # declared as a Factor Value or only as a Characteristic, so both
        # are considered and assay/ontology fields are excluded by name.
        if not col.startswith(("Factor Value", "Characteristics")):
            continue
        if NOT_PHENOTYPE.search(bracketed(col)) or NOT_PHENOTYPE.search(col):
            continue
        if not eligible_phenotype(col):
            continue
        vc = samples[col].dropna().astype(str).str.strip()
        vc = vc[vc.ne("") & vc.ne("nan")].value_counts()
        if len(vc) < 2:
            continue
        # a numeric factor with many levels is a covariate, not a group
        if len(vc) > 10 and pd.to_numeric(
                pd.Series(vc.index), errors="coerce").notna().all():
            continue
        l1, n1 = str(vc.index[0]), int(vc.iloc[0])
        l2, n2 = str(vc.index[1]), int(vc.iloc[1])
        if min(n1, n2) < MIN_PER_CLASS:
            continue
        out.append(Contrast(col, l1, l2, n1, n2))
    return out


# ------------------------------------------------------------------
def norm_name(s: str) -> str:
    """Canonical form for joining names across ISA-Tab files."""
    s = str(s).strip().lower()
    s = re.sub(r"\.(mzml|mzxml|raw|cdf|d|wiff|mzdata)$", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def maf_matrix(maf: pd.DataFrame,
               alias: dict[str, str]) -> pd.DataFrame | None:
    """Extract the samples x features matrix from a MAF table.

    Sample columns are identified by joining the MAF header against known
    aliases of each sample (sample name, assay name, raw file name)
    rather than by excluding known metadata columns, because the metadata
    column set is not fixed across submissions. Columns are returned
    under the sample-file name so downstream joins are unambiguous.
    """
    hits = {c: alias[norm_name(c)] for c in maf.columns
            if norm_name(c) in alias}
    if len(set(hits.values())) < MIN_SAMPLES:
        return None
    cols = list(hits)
    X = maf[cols].apply(pd.to_numeric, errors="coerce").T
    X.index = [hits[c] for c in cols]
    X = X[~X.index.duplicated(keep="first")]
    X = X.loc[:, X.notna().mean(axis=0) >= 0.5]
    if X.shape[1] < MIN_FEATURES:
        return None
    X = X.fillna(X.median(axis=0))
    keep = X.std(axis=0) > 1e-12
    X = X.loc[:, keep]
    return X if X.shape[1] >= MIN_FEATURES else None


def log_scale(X: pd.DataFrame) -> np.ndarray:
    """Log-transform intensities, guarding against non-positive values.

    Intensities are shifted by the smallest positive value rather than
    clipped at 1.0; clipping at an absolute threshold silently flattens
    features whose whole range lies below it.
    """
    V = X.to_numpy(dtype=float)
    V = np.where(np.isfinite(V), V, np.nan)
    pos = V[np.isfinite(V) & (V > 0)]
    if pos.size == 0:
        return V
    floor = pos.min()
    V = np.where(np.isfinite(V), np.maximum(V, floor), floor)
    return np.log2(V)
