"""Shared HTTP helpers for the MetaboLights screen.

Every request is cached on disk. The screen makes tens of thousands of
requests across several phases and is re-run as the qualifying criteria
are tightened, so nothing should be fetched from EBI twice.
"""
from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import requests

BASE = "https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public"
WS = "https://www.ebi.ac.uk/metabolights/ws"
ROOT = Path(__file__).parent
CACHE = ROOT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = 12
TIMEOUT = 45
RETRIES = 3

_session = requests.Session()
_session.headers["User-Agent"] = (
    "confound-screen/1.0 (academic re-analysis; contact via GitHub "
    "23008613g/C-LipoVAE)"
)


def _cache_path(key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", key)[:180]
    return CACHE / safe


def get(url: str, *, key: str, head_bytes: int | None = None) -> str | None:
    """Fetch a URL, caching the body. Returns None on permanent failure.

    head_bytes issues a Range request; servers that ignore Range simply
    return more data, which is harmless because we only parse the front.
    """
    p = _cache_path(key)
    if p.exists():
        return p.read_text(errors="replace")

    headers = {}
    if head_bytes:
        headers["Range"] = f"bytes=0-{head_bytes - 1}"

    for attempt in range(RETRIES):
        try:
            r = _session.get(url, timeout=TIMEOUT, headers=headers)
            if r.status_code in (200, 206):
                text = r.content.decode("utf-8", errors="replace")
                p.write_text(text)
                return text
            if r.status_code == 404:
                p.write_text("")            # cache the miss
                return ""
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def get_binary(url: str, *, key: str) -> Path | None:
    """Download to the cache and return the path, for large tables.

    The write is staged through a process-unique temporary file and
    renamed into place, so that several workers downloading in parallel
    can never observe a half-written table.
    """
    p = _cache_path(key)
    if p.exists():
        return p if p.stat().st_size else None
    tmp = p.with_suffix(p.suffix + f".{os.getpid()}.part")
    for attempt in range(RETRIES):
        try:
            r = _session.get(url, timeout=TIMEOUT * 3, stream=True)
            if r.status_code == 200:
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 16):
                        fh.write(chunk)
                os.replace(tmp, p)
                return p
            if r.status_code == 404:
                p.touch()
                return None
        except requests.RequestException:
            pass
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        time.sleep(1.5 * (attempt + 1))
    return None


def parallel(fn: Callable, items: Iterable, *, desc: str = "",
             workers: int = MAX_WORKERS) -> list:
    """Map fn over items with a progress line every 100 completions."""
    items = list(items)
    out: list = [None] * len(items)
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for f in futs:
            pass
        from concurrent.futures import as_completed
        for f in as_completed(futs):
            out[futs[f]] = f.result()
            done += 1
            if done % 100 == 0 or done == len(items):
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(items) - done) / rate if rate else 0
                print(f"  {desc} {done}/{len(items)}  "
                      f"{rate:.1f}/s  eta {eta/60:.1f} min", flush=True)
    return out


def list_dir(study: str) -> list[str]:
    """Filenames in a study's public directory."""
    html = get(f"{BASE}/{study}/", key=f"dir_{study}.html")
    if not html:
        return []
    return [h for h in re.findall(r'href="([^"?/][^"]*)"', html)
            if not h.startswith(("/", "http"))]
