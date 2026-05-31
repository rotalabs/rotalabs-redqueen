# Changelog

All notable changes to rotalabs-redqueen are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-05-31

Major release: a reproducible, conformance-gated engine that spans single-turn, multi-turn,
and agentic/MCP attack surfaces, with archive persistence and standards-aligned reporting.

### Added
- **Canonical PRNG** (`Rng`, xoshiro256++ seeded by SplitMix64): seedable, reproducible, and
  cross-language portable — cross-validated against an independent JavaScript reimplementation.
- **`Stimulus`** phenotype (`single_turn` / `multi_turn` / `agentic`) plus `Message`, `ToolCall`,
  and `Transcript` records.
- **`MultiTurnGenome`** — Crescendo-style escalation attacks evolved as scripted conversations.
- **`AgenticGenome`** + agentic `MockTarget` execution — multi-step tool-use / MCP exploit plans.
- **`LexicaseSelection`** and **`GeminiTarget`**.
- **Archive persistence**: `MapElitesArchive.save()` / `load()` / `seed()` for cross-run continuity.
- **Compliance reporting**: `ReportExporter` projects an archive over the attack taxonomy into an
  OWASP / MITRE ATLAS / EU AI Act Art. 55 / NIST AI RMF evidence report (`Report`, `TaxonomyLabel`).
- **Co-evolution** — `coevolve` (attacker vs defender), the `Defense` interface, a reference
  `SystemPromptDefense`, a defense-aware `MockTarget`, and `DefenderBlockFitness`.
- **Cross-model transfer measurement** via `MultiTargetFitness`.
- **Conformance suite** (L1 engine, L2 LLM domain, L3 report, L4 multi-turn & agentic,
  L5 co-evolution) gated by seeded golden fixtures, reproduced byte-for-byte by the TypeScript port.

### Changed (breaking)
- `Genome.to_phenotype() -> str` is now **`to_stimulus() -> Stimulus`**.
- `Genome` now requires **`to_dict()` / `from_dict()`** (lossless serialization for persistence).
- `LLMTarget.query(prompt)` is replaced by **`interact(stimulus) -> Transcript`**; providers
  implement `_complete(messages)`. `query()` remains as a single-prompt convenience.
- `Judge.judge(prompt, response)` is now **`judge(stimulus, transcript)`**; `JudgeResult` gains
  `success`, `confidence`, and `labels`.
- The engine uses the canonical `Rng` instead of `numpy.random.Generator`; seeded runs are now
  bit-reproducible. `MockTarget` is deterministic (a pure function of the prompt).

### Migration from 1.x
- `genome.to_phenotype()` → `genome.to_stimulus()`; implement `to_dict()` / `from_dict()`.
- `await target.query(prompt)` → `await target.interact(Stimulus.single_turn(prompt))`
  (or keep `query()` for a single prompt).
- Custom judges/fitness move to the `(stimulus, transcript)` signature.

## [1.0.0]

Initial Python release: core evolutionary framework (MAP-Elites, novelty, tournament) and the
single-turn LLM jailbreak domain.
