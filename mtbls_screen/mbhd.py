"""MicrobiomeHD as the fourth omics platform.

Duvallet et al. (Nat Commun 2017) reprocessed 28 case-control gut
microbiome studies through one pipeline and deposited the results on
Zenodo. Because every study went through the same processing, the
platform contributes datasets whose only heterogeneity is biological --
a useful contrast to Expression Atlas and MetaboLights, where
processing differs between submissions.

Per study archive:
    {study}.metadata.txt                        sample -> DiseaseState
    RDP/{study}.otu_table.100.denovo.rdp_assigned   taxa x samples
"""
from __future__ import annotations

import io
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ZENODO = "https://zenodo.org/api/records/569601"
ROOT = Path(__file__).parent
CACHE = ROOT / "cache_mbhd"
CACHE.mkdir(exist_ok=True)

MIN_SAMPLES = 30
MIN_PER_CLASS = 10
MIN_FEATURES = 20

# "H" marks the healthy controls throughout MicrobiomeHD; anything else
# is a case group, and studies with several case groups contribute the
# most frequent one so that exactly one contrast is taken per study.
HEALTHY = {"h", "healthy", "control", "nonibd", "non-ibd"}

_session = requests.Session()
_session.headers["User-Agent"] = "omics-protocol-benchmark/1.0 (academic)"


@dataclass(frozen=True)
class MbhdDataset:
    study: str
    X: np.ndarray
    y: np.ndarray
    factor: str
    levels: tuple[str, str]


def list_archives() -> list[dict]:
    r = _session.get(ZENODO, timeout=60)
    r.raise_for_status()
    return [{"key": f["key"], "url": f["links"]["self"],
             "size": f.get("size", 0)}
            for f in r.json().get("files", [])
            if f["key"].endswith("_results.tar.gz")]


def fetch(entry: dict) -> Path | None:
    p = CACHE / entry["key"]
    if p.exists() and p.stat().st_size:
        return p
    tmp = p.with_suffix(".part")
    try:
        with _session.get(entry["url"], timeout=600, stream=True) as r:
            if r.status_code != 200:
                return None
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 18):
                    fh.write(chunk)
        tmp.replace(p)
        return p
    except requests.RequestException:
        tmp.unlink(missing_ok=True)
        return None


def _member(tar: tarfile.TarFile, pattern: str) -> pd.DataFrame | None:
    """First archive member matching `pattern`, read as TSV.

    Two quirks of these archives are handled: they were packed on macOS,
    so AppleDouble `._` sidecars shadow the real files and must be
    skipped, and several metadata tables are Latin-1 rather than UTF-8.
    """
    rx = re.compile(pattern)
    for m in tar.getmembers():
        name = m.name.split("/")[-1]
        if not m.isfile() or name.startswith("._") or not rx.search(m.name):
            continue
        f = tar.extractfile(m)
        if f is None:
            continue
        raw = f.read()
        for enc in ("utf-8", "latin-1"):
            try:
                return pd.read_csv(io.BytesIO(raw), sep="\t", index_col=0,
                                   low_memory=False, encoding=enc)
            except (UnicodeDecodeError, ValueError):
                continue
            except Exception:
                return None
    return None


def load_archive(entry: dict) -> tuple[MbhdDataset | None, str]:
    p = fetch(entry)
    if p is None:
        return None, "download_failed"
    study = entry["key"].replace("_results.tar.gz", "")

    try:
        with tarfile.open(p, "r:gz") as tar:
            meta = _member(tar, r"\.metadata\.txt$")
            # `a or b` cannot be used here: a DataFrame has no truth
            # value, so the fallback must be an explicit None check
            otu = _member(tar, r"rdp_assigned$")
            if otu is None:
                otu = _member(tar, r"otu_table\.100\.denovo$")
    except Exception as exc:
        return None, f"archive_error: {type(exc).__name__}"
    if meta is None or otu is None:
        return None, "missing_tables"

    # Sample identifiers are numeric in some archives, so pandas infers
    # int for one table and str for the other and the join silently
    # yields nothing. Both sides are cast before matching.
    meta.index = meta.index.astype(str).str.strip()
    otu.columns = [str(c).strip() for c in otu.columns]

    col = next((c for c in meta.columns
                if c.lower() in ("diseasestate", "disease_state")), None)
    if col is None:
        return None, "no_disease_state"

    state = meta[col].astype(str).str.strip()
    healthy = state.str.lower().isin(HEALTHY)
    if healthy.sum() < MIN_PER_CLASS:
        return None, "no_control_group"
    cases = state[~healthy].value_counts()
    if not len(cases) or cases.iloc[0] < MIN_PER_CLASS:
        return None, "no_case_group"
    case = str(cases.index[0])

    keep = state[healthy | state.eq(case)]
    X = otu.T                                    # samples x taxa
    idx = [s for s in X.index if s in keep.index]
    if len(idx) < MIN_SAMPLES:
        return None, "too_few_aligned"

    y = (keep.loc[idx] == case).astype(int).to_numpy()
    if min(np.bincount(y, minlength=2)) < MIN_PER_CLASS:
        return None, "class_too_small"

    V = X.loc[idx].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    V = np.where(np.isfinite(V), V, 0.0)
    tot = V.sum(axis=1, keepdims=True)
    tot[tot <= 0] = 1.0
    # relative abundance, then log with a half-minimum pseudocount:
    # counts contain true zeros, so a fixed floor of 1 would distort the
    # low end of a compositional profile
    V = V / tot
    pos = V[V > 0]
    V = np.log2(V + (pos.min() / 2 if pos.size else 1e-6))

    keep_f = V.std(axis=0) > 1e-12
    V = V[:, keep_f]
    if V.shape[1] < MIN_FEATURES:
        return None, "too_few_features"

    return MbhdDataset(study=study, X=V, y=y, factor="DiseaseState",
                       levels=(case, "H")), "ok"
