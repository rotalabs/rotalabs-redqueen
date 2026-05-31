"""Compliance report projection (taxonomy -> OWASP / ATLAS / EU AI Act / NIST)."""

import asyncio

from rotalabs_redqueen import (
    AttackStrategy,
    BehaviorDimension,
    Encoding,
    JailbreakFitness,
    LLMAttackGenome,
    MapElitesArchive,
    MockTarget,
    Report,
    ReportExporter,
    evolve,
)


def _comply_archive(seed: int) -> MapElitesArchive:
    archive = MapElitesArchive(
        dimensions=[
            BehaviorDimension("strategy", 0.0, 1.0, len(AttackStrategy)),
            BehaviorDimension("encoding", 0.0, 1.0, len(Encoding)),
            BehaviorDimension("has_persona", 0.0, 1.0, 2),
        ]
    )
    asyncio.run(
        evolve(
            genome_class=LLMAttackGenome,
            fitness=JailbreakFitness(MockTarget(MockTarget.Mode.ALWAYS_COMPLY)),
            generations=8,
            population_size=16,
            seed=seed,
            archive=archive,
            progress=False,
        )
    )
    return archive


def test_report_projection():
    archive = _comply_archive(3)
    report = ReportExporter().export(
        archive.get_all(), campaign_id="t", coverage=archive.coverage()
    )
    assert isinstance(report, Report)
    d = report.to_dict()

    assert d["summary"]["attacks_found"] >= 1
    assert d["summary"]["coverage_percent"] > 0
    # successful single-turn attacks crosswalk to at least one OWASP entry
    assert len(d["standards"]["owasp"]) >= 1
    assert d["standards"]["eu_ai_act_art55"]["adversarial_testing_documented"] is True
    assert d["standards"]["nist_ai_rmf"]["govern_1_7"] == "addressed"
    assert len(d["evidence"]) == d["summary"]["attacks_found"]


def test_report_markdown_render():
    archive = _comply_archive(4)
    exporter = ReportExporter()
    report = exporter.export(archive.get_all(), campaign_id="md", coverage=archive.coverage())
    md = exporter.render(report, "markdown")
    assert b"Adversarial Testing Report" in md
    assert b"Standards coverage" in md


def test_empty_report_not_documented():
    empty = MapElitesArchive([BehaviorDimension("strategy", 0.0, 1.0, 6)])
    report = ReportExporter().export(empty.get_all(), campaign_id="empty")
    d = report.to_dict()
    assert d["summary"]["attacks_found"] == 0
    assert d["standards"]["eu_ai_act_art55"]["adversarial_testing_documented"] is False
