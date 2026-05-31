"""Conformance: seeded runs reproduce the golden fixtures byte-for-byte.

Golden fixtures live in ``tests/conformance/`` and are regenerated with
``python -m rotalabs_redqueen._gen_conformance`` (see that module). A failure
here means the engine's deterministic behavior changed -- intentional changes
must regenerate the goldens (a spec-version event).
"""

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

FIXTURES = Path(__file__).parent / "conformance"


def _assert_matches_golden(level: str, produced: dict):
    golden = (FIXTURES / f"{level}.json").read_text()
    assert canonical_json(produced) + "\n" == golden


def test_l1_engine_determinism():
    _assert_matches_golden("L1", run_l1(CONFORMANCE_SEED))


def test_l2_llm_determinism():
    _assert_matches_golden("L2", run_l2(CONFORMANCE_SEED))


def test_l3_report_projection():
    _assert_matches_golden("L3", run_l3(CONFORMANCE_SEED))


def test_l4_multiturn_determinism():
    _assert_matches_golden("L4_multiturn", run_l4_multiturn(CONFORMANCE_SEED))


def test_l4_agentic_determinism():
    _assert_matches_golden("L4_agentic", run_l4_agentic(CONFORMANCE_SEED))


def test_l5_coevolution_determinism():
    _assert_matches_golden("L5_coevolution", run_l5(CONFORMANCE_SEED))


def test_l2_is_reproducible_within_run():
    assert canonical_json(run_l2(CONFORMANCE_SEED)) == canonical_json(run_l2(CONFORMANCE_SEED))
