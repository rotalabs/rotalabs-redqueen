"""Canonical JSON for deterministic, cross-language comparison and signing.

Per redqueen-spec conformance.md §1.4: keys sorted, compact separators, UTF-8.
Numbers are formatted the ECMAScript / RFC 8785 way (integer-valued floats emit
without a trailing ``.0`` -- ``1.0`` -> ``1``), so a TypeScript or Rust port
using native JSON number formatting reproduces these bytes exactly. (Python's
default ``json.dumps`` would emit ``1.0``, which would not match JS.)
"""

from __future__ import annotations

import json
import math
from typing import Any


def _format_number(x: float | int) -> str:
    if isinstance(x, bool):  # bool is an int subclass; should be handled upstream
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if not math.isfinite(x):
        raise ValueError("non-finite numbers are not valid JSON")
    # Integer-valued floats serialize without a decimal point (ECMAScript ToString).
    if x == int(x) and abs(x) < 1e16:
        return str(int(x))
    return repr(x)  # shortest round-trip; matches ECMAScript for our value ranges


def _serialize(obj: Any, out: list[str]) -> None:
    if obj is True:
        out.append("true")
    elif obj is False:
        out.append("false")
    elif obj is None:
        out.append("null")
    elif isinstance(obj, str):
        out.append(json.dumps(obj, ensure_ascii=False))
    elif isinstance(obj, (int, float)):
        out.append(_format_number(obj))
    elif isinstance(obj, dict):
        out.append("{")
        for i, key in enumerate(sorted(obj)):
            if i:
                out.append(",")
            out.append(json.dumps(str(key), ensure_ascii=False))
            out.append(":")
            _serialize(obj[key], out)
        out.append("}")
    elif isinstance(obj, (list, tuple)):
        out.append("[")
        for i, value in enumerate(obj):
            if i:
                out.append(",")
            _serialize(value, out)
        out.append("]")
    else:
        raise TypeError(f"Cannot canonicalize value of type {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    """Serialize ``obj`` to canonical JSON (sorted keys, ECMAScript numbers)."""
    out: list[str] = []
    _serialize(obj, out)
    return "".join(out)
