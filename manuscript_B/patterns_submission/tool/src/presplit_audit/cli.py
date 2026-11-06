"""Command line: ``presplit-audit data.csv --label LABEL [--group GROUP]``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .permutation import audit
from .steps import BUILTIN


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="presplit-audit", description=(
        "Compare a classifier's cross-validated AUC when a supervised feature step is fitted to all samples "
        "(pre-split) and within each training fold (fold-contained)."))
    p.add_argument("data", type=Path, help="CSV with one row per sample")
    p.add_argument("--label", required=True, help="column holding the binary phenotype")
    p.add_argument("--group", help="column identifying participants, kept within one fold")
    p.add_argument("--drop", nargs="*", default=[], help="other non-feature columns to ignore")
    p.add_argument("--step", choices=sorted(BUILTIN), default="top_q")
    p.add_argument("--permutations", type=int, default=199)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--json", type=Path, help="write the full result, including null AUCs, to this file")
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    frame = pd.read_csv(args.data)
    meta = [args.label] + ([args.group] if args.group else []) + list(args.drop)
    missing = [c for c in meta if c not in frame.columns]
    if missing:
        print(f"error: columns not found: {missing}", file=sys.stderr)
        return 2
    features = frame.drop(columns=meta)
    non_numeric = [c for c in features.columns if not pd.api.types.is_numeric_dtype(features[c])]
    if non_numeric:
        print(f"error: non-numeric feature columns (list them with --drop): {non_numeric[:5]}", file=sys.stderr)
        return 2
    groups = frame[args.group].to_numpy() if args.group else None
    result = audit(features.to_numpy(dtype=float), frame[args.label].to_numpy(), args.step, groups=groups,
                   n_permutations=args.permutations, random_state=args.seed, n_jobs=args.jobs)
    print(result.summary())
    if args.json:
        args.json.write_text(json.dumps(result.to_dict(), indent=2, default=lambda v: np.asarray(v).tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
