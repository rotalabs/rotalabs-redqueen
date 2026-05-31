"""Regenerate conformance golden fixtures.

Usage:
    python -m rotalabs_redqueen._gen_conformance [output_dir]

``output_dir`` defaults to ``tests/conformance``. Run this only when a change to
deterministic behavior is intentional; committing the regenerated fixtures is a
spec-version event (redqueen-spec conformance.md §5).
"""

import sys
from pathlib import Path

from rotalabs_redqueen.conformance import (
    CONFORMANCE_SEED,
    run_l1,
    run_l2,
    run_l3,
    run_l4_agentic,
    run_l4_multiturn,
    run_l5,
)
from rotalabs_redqueen.core.canonical import canonical_json


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/conformance")
    out.mkdir(parents=True, exist_ok=True)
    levels = (
        ("L1", run_l1),
        ("L2", run_l2),
        ("L3", run_l3),
        ("L4_multiturn", run_l4_multiturn),
        ("L4_agentic", run_l4_agentic),
        ("L5_coevolution", run_l5),
    )
    for level, fn in levels:
        path = out / f"{level}.json"
        path.write_text(canonical_json(fn(CONFORMANCE_SEED)) + "\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
