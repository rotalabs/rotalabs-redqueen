"""Compliance report: a projection of a quality-diversity archive over the taxonomy.

This is the artifact that turns "we ran some attacks" into documented,
standards-aligned evidence (redqueen-spec types.md §Report). It groups successful
attacks by harm category and crosswalks them to OWASP / MITRE ATLAS / EU AI Act /
NIST AI RMF, using each genome's ``label()``.
"""

from __future__ import annotations

from dataclasses import dataclass

from rotalabs_redqueen.core.canonical import canonical_json
from rotalabs_redqueen.core.stimulus import SPEC_VERSION
from rotalabs_redqueen.core.taxonomy import atlas_ids, owasp_ids

REPORT_VERSION = "0.1.0"


@dataclass
class Report:
    """A compliance/evidence report (types.md §Report)."""

    campaign_id: str
    summary: dict
    by_harm_category: list
    standards: dict
    evidence: list
    spec_version: str = SPEC_VERSION
    report_version: str = REPORT_VERSION

    def to_dict(self) -> dict:
        return {
            "spec_version": self.spec_version,
            "report_version": self.report_version,
            "campaign_id": self.campaign_id,
            "summary": self.summary,
            "by_harm_category": self.by_harm_category,
            "standards": self.standards,
            "evidence": self.evidence,
        }


class ReportExporter:
    """Project archive elites into a :class:`Report` (redqueen-spec interfaces.md §6).

    An attack counts as successful when its fitness >= ``success_threshold``.
    Each genome must implement ``label() -> TaxonomyLabel``.
    """

    def __init__(self, success_threshold: float = 0.5):
        self.success_threshold = success_threshold

    def export(self, individuals, *, campaign_id: str = "campaign", coverage=None) -> Report:
        individuals = list(individuals)
        successful = [i for i in individuals if i.fitness.value >= self.success_threshold]

        harm: dict[str, dict] = {}
        owasp_counts: dict[str, int] = {}
        atlas_counts: dict[str, int] = {}
        evidence = []

        for ind in successful:
            label = ind.genome.label()
            score = ind.fitness.value
            bucket = harm.setdefault(label.harm_category, {"count": 0, "max_score": 0.0})
            bucket["count"] += 1
            bucket["max_score"] = max(bucket["max_score"], score)
            for oid in owasp_ids(label.strategy):
                owasp_counts[oid] = owasp_counts.get(oid, 0) + 1
            for aid in atlas_ids(label.strategy):
                atlas_counts[aid] = atlas_counts.get(aid, 0) + 1
            evidence.append({"label": label.to_dict(), "score": score})

        summary = {
            "evaluated": len(individuals),
            "attacks_found": len(successful),
            "success_threshold": self.success_threshold,
        }
        if coverage is not None:
            summary["coverage_percent"] = coverage.coverage_percent
            summary["filled_cells"] = coverage.filled_cells

        by_harm_category = [
            {"harm_category": h, "count": v["count"], "max_score": v["max_score"]}
            for h, v in sorted(harm.items())
        ]

        standards = {
            "owasp": [
                {"id": k, "covered": True, "evidence_count": v}
                for k, v in sorted(owasp_counts.items())
            ],
            "mitre_atlas": [
                {"id": k, "covered": True, "evidence_count": v}
                for k, v in sorted(atlas_counts.items())
            ],
            "eu_ai_act_art55": {"adversarial_testing_documented": len(successful) > 0},
            "nist_ai_rmf": {"govern_1_7": "addressed" if len(successful) > 0 else "not_addressed"},
        }

        return Report(
            campaign_id=campaign_id,
            summary=summary,
            by_harm_category=by_harm_category,
            standards=standards,
            evidence=evidence,
        )

    def render(self, report: Report, fmt: str = "json") -> bytes:
        """Render a report to ``json`` or ``markdown`` bytes."""
        if fmt == "json":
            return (canonical_json(report.to_dict()) + "\n").encode()
        if fmt == "markdown":
            return self._markdown(report).encode()
        raise ValueError(f"Unknown report format: {fmt}")

    def _markdown(self, report: Report) -> str:
        s = report.summary
        lines = [
            f"# Adversarial Testing Report — {report.campaign_id}",
            "",
            f"- Attacks found: **{s['attacks_found']}** of {s['evaluated']} evaluated",
        ]
        if "coverage_percent" in s:
            lines.append(
                f"- Archive coverage: **{s['coverage_percent']:.1f}%** ({s['filled_cells']} cells)"
            )
        lines += ["", "## By harm category", ""]
        for row in report.by_harm_category:
            lines.append(
                f"- `{row['harm_category']}`: {row['count']} (max score {row['max_score']:.2f})"
            )
        lines += ["", "## Standards coverage", ""]
        for oid in report.standards["owasp"]:
            lines.append(f"- OWASP `{oid['id']}` — {oid['evidence_count']} evidence item(s)")
        for aid in report.standards["mitre_atlas"]:
            lines.append(f"- MITRE ATLAS `{aid['id']}` — {aid['evidence_count']} evidence item(s)")
        art55 = report.standards["eu_ai_act_art55"]["adversarial_testing_documented"]
        lines += [
            "",
            f"- EU AI Act Art. 55 adversarial testing documented: **{art55}**",
            f"- NIST AI RMF GOVERN 1.7: **{report.standards['nist_ai_rmf']['govern_1_7']}**",
            "",
        ]
        return "\n".join(lines)
