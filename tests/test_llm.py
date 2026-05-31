"""Tests for LLM adversarial testing domain."""

import pytest

from rotalabs_redqueen import Message, Rng, Stimulus, Transcript, evolve
from rotalabs_redqueen.llm import (
    AgenticGenome,
    AgenticStrategy,
    AttackStrategy,
    Encoding,
    Escalation,
    GeminiTarget,
    HarmCategory,
    HeuristicJudge,
    JailbreakFitness,
    LLMAttackGenome,
    MockTarget,
    MultiTurnGenome,
    Persona,
    create_target,
)


class TestLLMAttackGenome:
    """Tests for LLMAttackGenome."""

    def test_random_creation(self):
        rng = Rng(42)
        genome = LLMAttackGenome.random(rng)
        assert genome.primary_strategy in AttackStrategy
        assert genome.encoding in Encoding
        assert genome.target_category in HarmCategory

    def test_to_prompt(self):
        genome = LLMAttackGenome(
            primary_strategy=AttackStrategy.ROLEPLAY,
            persona=Persona("TestBot", "A test persona"),
            core="Explain something",
        )
        prompt = genome.to_prompt()
        assert "TestBot" in prompt
        assert "Explain something" in prompt

    def test_mutation(self):
        rng = Rng(42)
        original = LLMAttackGenome.random(rng)
        mutated = original.mutate(rng)
        # Should be different instance
        assert mutated is not original

    def test_crossover(self):
        rng = Rng(42)
        parent1 = LLMAttackGenome.random(rng)
        parent2 = LLMAttackGenome.random(rng)
        child = parent1.crossover(parent2, rng)
        # Child should have genes from either parent
        assert child.primary_strategy in [parent1.primary_strategy, parent2.primary_strategy]

    def test_encoding_base64(self):
        genome = LLMAttackGenome(
            primary_strategy=AttackStrategy.ENCODING,
            encoding=Encoding.BASE64,
            core="test",
        )
        prompt = genome.to_prompt()
        # "test" in base64 is "dGVzdA=="
        assert "dGVzdA==" in prompt

    def test_encoding_rot13(self):
        genome = LLMAttackGenome(
            primary_strategy=AttackStrategy.ENCODING,
            encoding=Encoding.ROT13,
            core="test",
        )
        prompt = genome.to_prompt()
        # "test" in rot13 is "grfg"
        assert "grfg" in prompt

    def test_behavior(self):
        genome = LLMAttackGenome(
            primary_strategy=AttackStrategy.ROLEPLAY,
            encoding=Encoding.NONE,
            persona=Persona.PERSONAS[0],
        )
        bd = genome.behavior()
        assert len(bd.values) == 3
        assert 0 <= bd.values[0] <= 1
        assert 0 <= bd.values[1] <= 1
        assert bd.values[2] == 1.0  # Has persona


class TestMockTarget:
    """Tests for MockTarget."""

    @pytest.mark.asyncio
    async def test_always_refuse(self):
        target = MockTarget(MockTarget.Mode.ALWAYS_REFUSE)
        response = await target.query("test")
        assert "cannot" in response.content.lower()

    @pytest.mark.asyncio
    async def test_always_comply(self):
        target = MockTarget(MockTarget.Mode.ALWAYS_COMPLY)
        response = await target.query("test")
        assert "sure" in response.content.lower() or "here" in response.content.lower()

    @pytest.mark.asyncio
    async def test_random_mode(self):
        target = MockTarget(MockTarget.Mode.RANDOM)
        # Should not raise
        response = await target.query("test")
        assert len(response.content) > 0


class TestCreateTarget:
    """Tests for create_target factory."""

    def test_mock_target(self):
        target = create_target("mock:random")
        assert target.name == "mock:random"

    def test_mock_refuse(self):
        target = create_target("mock:refuse")
        assert target.name == "mock:refuse"

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_target("unknown:model")


def _heard(response: str) -> Transcript:
    """Build a single-turn transcript whose assistant reply is ``response``."""
    return Transcript(
        target_id="mock",
        stimulus_kind="single_turn",
        messages=[Message("user", "test"), Message("assistant", response)],
    )


class TestHeuristicJudge:
    """Tests for HeuristicJudge."""

    @pytest.mark.asyncio
    async def test_refusal_detection(self):
        result = await HeuristicJudge().judge(
            Stimulus.single_turn("test"),
            _heard("I cannot and will not help with that request."),
        )
        assert result.refused
        assert result.score < 0.5

    @pytest.mark.asyncio
    async def test_compliance_detection(self):
        result = await HeuristicJudge().judge(
            Stimulus.single_turn("test"),
            _heard("Sure, here is how you do it. Step 1: First, you need to..."),
        )
        assert not result.refused
        assert result.score > 0.5

    @pytest.mark.asyncio
    async def test_mixed_signals(self):
        result = await HeuristicJudge().judge(
            Stimulus.single_turn("test"),
            _heard("I cannot help with harmful requests, but here is some general info..."),
        )
        # Mixed signals should give intermediate score
        assert 0 < result.score < 1


class TestJailbreakFitness:
    """Tests for JailbreakFitness."""

    @pytest.mark.asyncio
    async def test_evaluate_refuse_target(self):
        target = MockTarget(MockTarget.Mode.ALWAYS_REFUSE)
        fitness = JailbreakFitness(target)
        genome = LLMAttackGenome.random()

        result = await fitness.evaluate(genome)
        # Refusal should give low fitness
        assert result.fitness.value < 0.5

    @pytest.mark.asyncio
    async def test_evaluate_comply_target(self):
        target = MockTarget(MockTarget.Mode.ALWAYS_COMPLY)
        fitness = JailbreakFitness(target)
        genome = LLMAttackGenome.random()

        result = await fitness.evaluate(genome)
        # Compliance should give high fitness
        assert result.fitness.value > 0.5

    @pytest.mark.asyncio
    async def test_evaluate_batch(self):
        target = MockTarget()
        fitness = JailbreakFitness(target)
        genomes = [LLMAttackGenome.random() for _ in range(5)]

        results = await fitness.evaluate_batch(genomes)
        assert len(results) == 5


class TestGeminiTarget:
    """Tests for GeminiTarget (construction + metadata only; no network)."""

    def test_name_id_and_kinds(self):
        target = GeminiTarget(api_key="test-key")
        assert target.name == "gemini:gemini-2.0-flash"
        assert target.id == target.name
        assert "single_turn" in target.supported_kinds

    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Gemini API key required"):
            GeminiTarget()


class TestMultiTurnGenome:
    """Tests for the multi-turn (Crescendo-style) genome."""

    def test_to_stimulus_shape(self):
        g = MultiTurnGenome(num_turns=4, escalation=Escalation.RESEARCH_FRAME)
        stim = g.to_stimulus()
        assert stim.kind == "multi_turn"
        assert len([m for m in stim.turns if m.role == "user"]) == 4
        assert g.core in stim.turns[-1].content  # final turn is the core ask

    def test_random_behavior_bounds(self):
        bd = MultiTurnGenome.random(Rng(3)).behavior()
        assert len(bd.values) == 3
        assert all(0.0 <= v <= 1.0 for v in bd.values)

    @pytest.mark.asyncio
    async def test_fitness_runs(self):
        fitness = JailbreakFitness(MockTarget(MockTarget.Mode.ALWAYS_COMPLY))
        result = await fitness.evaluate(MultiTurnGenome.random(Rng(1)))
        assert result.fitness.value > 0.5


class TestAgenticGenome:
    """Tests for the basic agentic / tool-use genome."""

    def test_to_stimulus_shape(self):
        g = AgenticGenome(strategy=AgenticStrategy.GOAL_HIJACK, num_steps=3)
        stim = g.to_stimulus()
        assert stim.kind == "agentic"
        assert len(stim.action_plan) == 3
        assert stim.available_tools

    @pytest.mark.asyncio
    async def test_mock_executes_agentic(self):
        target = MockTarget(MockTarget.Mode.ALWAYS_COMPLY)
        transcript = await target.interact(AgenticGenome.random(Rng(2)).to_stimulus())
        assert transcript.stimulus_kind == "agentic"
        assert len(transcript.tool_calls) >= 1

    @pytest.mark.asyncio
    async def test_fitness_runs(self):
        fitness = JailbreakFitness(MockTarget(MockTarget.Mode.ALWAYS_COMPLY))
        result = await fitness.evaluate(AgenticGenome.random(Rng(5)))
        assert result.fitness.value > 0.5


@pytest.mark.asyncio
async def test_evolve_multiturn_and_agentic():
    """Both new genome types drive the generic evolution loop end-to-end."""
    for genome_cls in (MultiTurnGenome, AgenticGenome):
        fitness = JailbreakFitness(MockTarget())
        res = await evolve(
            genome_class=genome_cls,
            fitness=fitness,
            generations=5,
            population_size=8,
            seed=42,
            progress=False,
        )
        assert res.best is not None
