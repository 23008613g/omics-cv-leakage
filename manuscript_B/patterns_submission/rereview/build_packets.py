"""Evidence packets for a blinded re-review of the 258 unresolved human-subset datasets.

Each packet carries repository metadata only: the columns shown on the human
re-review sheet, plus study title, description and sample annotations from the
repository (local cache first; public EBI and Metabolomics Workbench APIs for
text the cache lacks). Columns holding the original reviews, adjudications or
automatic-rule outputs are never read (``usecols``), so the re-review is blind
to them. Network responses are cached under ``fetch_cache/``.
"""
from __future__ import annotations

import json
import re
import subprocess
import tarfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MANUSCRIPT_B = HERE.parents[1]
QUEUE = MANUSCRIPT_B / "Formal_Public_Deposit_v3" / "tables" / "human_cohort_adjudication_v2_manual.csv"
RUNNER = MANUSCRIPT_B.parent / "mtbls_screen"
CACHE, MBHD = RUNNER / "cache", RUNNER / "cache_mbhd"
FETCH = HERE / "fetch_cache"
SHOWN = ["id", "repository", "platform", "phenotype", "levels", "n_samples", "n_pos", "n_neg", "species_evidence",
         "evidence_url"]
QUEUE_FLAG = "reviewer_1_decision"          # read only to identify the 258 queued rows, never stored
SKIP_COLS = re.compile(r"(?i)(file|uri|accession|scan|date|ref|protocol ref|term source|performer|md5|raw|"
                       r"label|technology|assay name|extract name|hybridization|array design)")


def _get(url: str, name: str) -> str:
    """Fetch with curl (this Python build has no CA bundle configured) and cache the response."""
    FETCH.mkdir(exist_ok=True)
    path = FETCH / name
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        subprocess.run(["curl", "-sSfL", "-m", "60", "-A", "presplit-audit-rereview/0.1", "-o", str(tmp), url],
                       check=True)
        tmp.rename(path)
    return path.read_text(errors="replace")


def _read_tsv(path) -> pd.DataFrame:
    """Repository TSVs are sometimes ragged; skip malformed lines rather than fail."""
    return pd.read_csv(path, sep="\t", dtype=str, on_bad_lines="skip", engine="python", encoding_errors="replace")


def _summarise(frame: pd.DataFrame, max_cols: int = 14, max_vals: int = 6) -> list[str]:
    out = []
    for c in frame.columns:
        if SKIP_COLS.search(str(c)) or len(out) >= max_cols:
            continue
        vals = frame[c].dropna().astype(str).str.strip()
        vals = vals[vals.ne("") & vals.ne("not applicable") & vals.ne("not available")]
        if vals.empty:
            continue
        u = vals.value_counts()
        shown = "; ".join(f"{k} ({v})" for k, v in u.head(max_vals).items())
        out.append(f"{c}: {shown}" + (f"; +{len(u) - max_vals} more" if len(u) > max_vals else ""))
    return out


def gxa(acc: str, catalogue: dict) -> dict:
    e = catalogue.get(acc, {})
    sdrf = CACHE / f"gxa_sdrf_{acc}.tsv"
    ann = _summarise(_read_tsv(sdrf)) if sdrf.exists() else ["(no SDRF in cache)"]
    return {"title": e.get("experimentDescription", ""), "species": e.get("species", ""),
            "type": e.get("experimentType", ""), "factors": e.get("experimentalFactors", []), "samples": ann}


def mw(st: str) -> dict:
    text = _get(f"https://www.metabolomicsworkbench.org/rest/study/study_id/{st}/mwtab/txt", f"{st}.mwtab.txt")
    keep = ("ST:STUDY_TITLE", "ST:STUDY_SUMMARY", "SU:SUBJECT_TYPE", "SU:TAXONOMY_ID", "SU:SUBJECT_SPECIES",
            "SU:CELL_STRAIN_DETAILS", "SU:CELL_BIOSOURCE_OR_SUPPLIER", "SU:SUBJECT_SUMMARY", "CO:COLLECTION_SUMMARY",
            "CO:SAMPLE_TYPE", "TR:TREATMENT_SUMMARY")
    fields: dict[str, list[str]] = {}
    for line in text.splitlines():
        key = line[:33].strip()
        if key in keep:
            fields.setdefault(key, []).append(line[33:].strip())
    fac = [ln.split("\t")[-1] for ln in text.splitlines() if ln.startswith("SUBJECT_SAMPLE_FACTORS")]
    levels = pd.Series(fac).value_counts().head(8).to_dict() if fac else {}
    return {k: " ".join(v)[:900] for k, v in fields.items()} | {"sample_factors": levels}


def mtbls(acc: str) -> dict:
    inv = next(iter(sorted(CACHE.glob(f"inv_{acc}*"))), None)
    info = {}
    if inv is not None:
        for line in inv.read_text(errors="replace").splitlines():
            key, _, rest = line.partition("\t")
            if key in ("Study Title", "Study Description", "Study Design Type", "Study Factor Name"):
                info[key] = re.sub(r"<[^>]+>", " ", rest.replace('"', "")).strip()[:1200]
    s = next(iter(sorted(CACHE.glob(f"f_{acc}_s_*"))), None)
    info["samples"] = _summarise(_read_tsv(s)) if s is not None else ["(no sample file)"]
    return info


def mbhd(study: str) -> dict:
    tars = sorted(MBHD.glob(f"{study}*results.tar.gz"))
    if not tars:
        return {"note": "no cached MicrobiomeHD archive"}
    with tarfile.open(tars[0], "r:gz") as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith(".metadata.txt")
                       and not m.name.split("/")[-1].startswith("._")), None)
        meta = pd.read_csv(tar.extractfile(member), sep="\t", dtype=str, encoding_errors="replace") if member else None
    return {"collection": "MicrobiomeHD (Duvallet et al. 2017): standardized 16S case-control gut microbiome studies",
            "samples": _summarise(meta) if meta is not None else ["(no metadata member)"]}


def main() -> None:
    flag = pd.read_csv(QUEUE, usecols=["id", QUEUE_FLAG])
    queued = set(flag.loc[flag[QUEUE_FLAG].notna(), "id"])
    del flag
    q = pd.read_csv(QUEUE, usecols=SHOWN)
    q = q[q.id.isin(queued)].sort_values("id").reset_index(drop=True)
    assert len(q) == 258, len(q)
    cat = {e["experimentAccession"]: e for e in json.loads(_get("https://www.ebi.ac.uk/gxa/json/experiments",
                                                               "gxa_experiments.json"))["experiments"]}
    packets = []
    for r in q.to_dict("records"):
        repo = r["repository"]
        extra = (gxa(r["id"], cat) if repo == "gxa" else mw(r["id"]) if repo == "mw"
                 else mtbls(r["id"]) if repo == "mtbls" else mbhd(r["id"]))
        packets.append({**{k: (None if pd.isna(v) else v) for k, v in r.items()}, "repository_text": extra})
    (HERE / "packets.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in packets) + "\n")
    text = json.dumps(packets, ensure_ascii=False)
    assert not re.search(r"(?i)reviewer_|adjudicat|manual_final|_auto\b", text), "a decision column leaked"
    print(f"{len(packets)} packets; repositories {q.repository.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
