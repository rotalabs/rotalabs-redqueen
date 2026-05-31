"""Conformance: the canonical PRNG reproduces the cross-validated spec vectors.

`prng_vectors.json` is the byte-for-byte copy of
`_dev/redqueen-spec/conformance/prng/vectors.json`, which is independently
reproduced by a JavaScript (BigInt) implementation. If this test passes, the
Python `Rng` agrees with both the reference oracle and the JS port.
"""

import json
from pathlib import Path

from rotalabs_redqueen.core.rng import Rng

VECTORS = json.loads((Path(__file__).parent / "prng_vectors.json").read_text())
_MASK = (1 << 64) - 1


def _hx(v: int) -> str:
    return "0x" + format(v & _MASK, "016x")


def test_seeding_and_u64_stream():
    for case in VECTORS["cases"]:
        seed = case["seed_decimal"]

        g = Rng(seed)
        assert [_hx(x) for x in g.s] == case["initial_state"]

        g = Rng(seed)
        assert [_hx(g.next_u64()) for _ in range(12)] == case["next_u64"]


def test_double_stream():
    for case in VECTORS["cases"]:
        g = Rng(case["seed_decimal"])
        got = [repr(g.next_double()) for _ in range(8)]
        assert got == [d["value"] for d in case["next_double"]]


def test_below_cases():
    for bc in VECTORS["below_cases"]:
        g = Rng(int(bc["seed"], 16))
        assert [g.below(bc["n"]) for _ in range(10)] == bc["outputs"]


def test_shuffle_cases():
    for sc in VECTORS["shuffle_cases"]:
        g = Rng(int(sc["seed"], 16))
        arr = list(sc["input"])
        g.shuffle(arr)
        assert arr == sc["output"]
