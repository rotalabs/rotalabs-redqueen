"""Attack taxonomy labels and standards crosswalk.

The taxonomy is the join key between attacks, archive cells, and compliance
reports (redqueen-spec taxonomy.md). A ``TaxonomyLabel`` describes one attack;
the crosswalk maps its strategy to external framework IDs (OWASP, MITRE ATLAS).
"""

from __future__ import annotations

from dataclasses import dataclass

TAXONOMY_VERSION = "2026.1"


@dataclass
class TaxonomyLabel:
    """A label placing one attack in the taxonomy (taxonomy.md §TaxonomyLabel)."""

    surface: str  # single_turn | multi_turn | agentic | mcp
    strategy: str  # taxonomy.md §Strategy
    harm_category: str  # taxonomy.md §HarmCategory
    encoding: str = "none"
    taxonomy_version: str = TAXONOMY_VERSION

    def to_dict(self) -> dict:
        return {
            "surface": self.surface,
            "strategy": self.strategy,
            "harm_category": self.harm_category,
            "encoding": self.encoding,
            "owasp": owasp_ids(self.strategy),
            "atlas": atlas_ids(self.strategy),
            "taxonomy_version": self.taxonomy_version,
        }


# Strategy -> OWASP IDs. LLM Top-10 (2025) IDs; Agentic Top-10 (2026) IDs prefixed
# AGENT* are PLACEHOLDERS pending reconciliation with the published OWASP list.
_OWASP: dict[str, list[str]] = {
    "direct": ["LLM01"],
    "roleplay": ["LLM01"],
    "authority": ["LLM01"],
    "hypothetical": ["LLM01"],
    "encoding": ["LLM01"],
    "prompt_injection": ["LLM01"],
    "multi_turn_escalation": ["LLM01"],
    "tool_misuse": ["LLM06", "AGENT02"],
    "goal_hijack": ["LLM06", "AGENT01"],
    "memory_poisoning": ["AGENT06"],
    "context_poisoning": ["AGENT06"],
}

# Strategy -> MITRE ATLAS technique IDs. PLACEHOLDERS — verify against the live
# ATLAS matrix before treating as authoritative (redqueen-spec taxonomy.md gate).
_ATLAS: dict[str, list[str]] = {
    "roleplay": ["AML.T0054"],
    "hypothetical": ["AML.T0054"],
    "multi_turn_escalation": ["AML.T0054"],
    "prompt_injection": ["AML.T0051"],
    "encoding": ["AML.T0051"],
    "tool_misuse": ["AML.T0053"],
    "goal_hijack": ["AML.T0053"],
}


def owasp_ids(strategy: str) -> list[str]:
    """OWASP Top-10 IDs a strategy maps to."""
    return list(_OWASP.get(strategy, []))


def atlas_ids(strategy: str) -> list[str]:
    """MITRE ATLAS technique IDs a strategy maps to (placeholders; see module note)."""
    return list(_ATLAS.get(strategy, []))
