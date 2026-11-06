"""Survey of the analysed Expression Atlas datasets for cross-study pairs (step 1 of the transcriptomic
cross-study validation; exploratory). For each dataset: species, technology and source matrix (array design),
the contrast chosen by the formal loader, and the organism parts of the included assays (condensed SDRF, local
cache of the formal run). Output: results/gxa_survey.json.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "validation"))
import validate_against_formal as V                       # noqa: E402
from cv_folds_all import analysed                         # noqa: E402

OUT = HERE / "results"


def survey(job) -> dict:
    from nm_v4 import raw_loaders as R
    dataset_id, platform, entry = job
    config, ds, X, y = V._load(dataset_id, platform, entry)
    md = dict(ds.metadata)
    sdrf = R.gxa_loader.read_sdrf(dataset_id)
    char = sdrf[sdrf["kind"].str.lower().isin(["characteristic", "factor"])]
    part = char[char["name"].str.lower().eq("organism part")].drop_duplicates("assay").set_index("assay")["value"]
    parts = Counter(str(part.get(s, "")) for s in ds.sample_ids)
    design = re.search(r"_(A-[A-Z]+-\d+)-", md["source_file"])
    return {"dataset_id": dataset_id, "species": md["species"], "assay": md["assay"],
            "source_file": md["source_file"], "array_design": design.group(1) if design else None,
            "transform_kind": ds.transform_kind, "phenotype": md["phenotype"], "level_pos": md["levels"][0],
            "level_neg": md["levels"][1], "n": int(y.size), "n_pos": int(y.sum()), "p": int(X.shape[1]),
            "feature_prefix": str(ds.feature_ids[0])[:4], "organism_parts": dict(parts.most_common())}


def main() -> int:
    from nm_v4 import raw_loaders
    t = analysed()
    ids = list(t[t.repository == "gxa"].dataset_id)
    jobs = [(i, p, dict(e)) for i, (p, e) in zip(ids, raw_loaders.entries_for_whitelist(ids), strict=True)]
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(survey, jobs))
    (OUT / "gxa_survey.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(len(rows), "datasets surveyed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
