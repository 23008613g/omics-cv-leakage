"""Shared scalar contracts for the independent QC validation pass."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
import re
from typing import Any


HASH = re.compile(r"^[0-9a-f]{64}$")


class QCError(ValueError):
    """The run or its audit tables cannot support a safe QC decision."""


def _fail(context: str, message: str) -> QCError:
    return QCError(f"{context}: {message}")


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    try:
        number = int(str(value))
    except (TypeError, ValueError) as exc:
        raise _fail(context, f"expected an integer >= {minimum}") from exc
    if str(value).strip() not in {str(number), f"{number}.0"} or number < minimum:
        raise _fail(context, f"expected an integer >= {minimum}")
    return number


def _number(value: Any, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise _fail(context, "expected a finite number") from exc
    if not math.isfinite(number):
        raise _fail(context, "expected a finite number")
    return number


def _boolean(value: Any, context: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise _fail(context, "expected a boolean")


def _hash(value: Any, context: str, *, nullable: bool = False) -> str | None:
    if (value is None or str(value).strip() == "") and nullable:
        return None
    normalized = str(value)
    if not HASH.fullmatch(normalized):
        raise _fail(context, "expected a lowercase SHA256")
    return normalized


def _json_cell(value: str, context: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise _fail(context, "invalid JSON cell") from exc


def _blank(value: Any) -> bool:
    return value is None or not str(value).strip()


def _rows_for(
    tables: Mapping[str, list[dict[str, Any]]],
    name: str,
    dataset: str,
) -> list[dict[str, Any]]:
    return [row for row in tables[name] if row["dataset"] == dataset]


__all__ = ["QCError"]
