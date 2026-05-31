"""LLM target adapters for adversarial testing.

A Target executes a `Stimulus` against a system under test and returns a
replayable `Transcript` (redqueen-spec interfaces.md §2). Providers implement
`_complete(messages) -> TargetResponse`; the base class turns that into a
single-turn or scripted multi-turn rollout. Agentic execution is handled by
agentic/MCP targets (added with the agentic genome).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import httpx

from rotalabs_redqueen.core.stimulus import (
    AGENTIC,
    MULTI_TURN,
    SINGLE_TURN,
    STOP_COMPLETED,
    STOP_MAX_TURNS,
    Message,
    Stimulus,
    ToolCall,
    Transcript,
)


class TargetError(Exception):
    """Error from LLM target."""


class RateLimitError(TargetError):
    """Rate limit exceeded."""


class NetworkError(TargetError):
    """Network connectivity error."""


@dataclass
class TargetResponse:
    """A single completion from a provider (internal to a Target)."""

    content: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0


class LLMTarget(ABC):
    """Abstract base class for LLM targets.

    Subclasses implement `name` and `_complete`. `interact` and `query` are
    provided by the base class.
    """

    # Kinds this target can execute. Override in agentic/MCP targets.
    supported_kinds: frozenset[str] = frozenset({SINGLE_TURN, MULTI_TURN})

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name, e.g. ``openai:gpt-4``."""

    @property
    def id(self) -> str:
        """Stable target id (redqueen-spec). Defaults to :attr:`name`."""
        return self.name

    @abstractmethod
    async def _complete(self, messages: list[Message]) -> TargetResponse:
        """Produce one assistant completion for a message list."""

    async def aclose(self) -> None:  # noqa: B027 - optional no-op hook, intentionally concrete
        """Release any resources (no-op by default)."""

    async def query(self, prompt: str) -> TargetResponse:
        """Convenience: complete a single user prompt."""
        return await self._complete([Message(role="user", content=prompt)])

    async def interact(self, stimulus: Stimulus) -> Transcript:
        """Execute a Stimulus and return a Transcript."""
        if stimulus.kind == SINGLE_TURN:
            messages: list[Message] = []
            if stimulus.system:
                messages.append(Message(role="system", content=stimulus.system))
            messages.append(Message(role="user", content=stimulus.prompt or ""))
            resp = await self._complete(messages)
            messages.append(Message(role="assistant", content=resp.content))
            return self._transcript(stimulus.kind, messages, resp, STOP_COMPLETED)

        if stimulus.kind == MULTI_TURN and (stimulus.mode or "scripted") == "scripted":
            messages = []
            last: TargetResponse | None = None
            turns = stimulus.turns or []
            stop = STOP_COMPLETED
            for turn in turns:
                messages.append(turn)
                if turn.role == "user":
                    if sum(1 for m in messages if m.role == "user") > stimulus.max_turns:
                        stop = STOP_MAX_TURNS
                        break
                    last = await self._complete(messages)
                    messages.append(Message(role="assistant", content=last.content))
            return self._transcript(stimulus.kind, messages, last, stop)

        raise TargetError(
            f"{self.id} does not support stimulus kind '{stimulus.kind}'"
            f" (supported: {sorted(self.supported_kinds)})"
        )

    def _transcript(
        self,
        kind: str,
        messages: list[Message],
        resp: TargetResponse | None,
        stop_reason: str,
    ) -> Transcript:
        return Transcript(
            target_id=self.id,
            stimulus_kind=kind,
            messages=messages,
            stop_reason=stop_reason,
            usage={
                "input_tokens": 0,
                "output_tokens": resp.tokens_used if resp else 0,
                "wall_ms": resp.latency_ms if resp else 0,
            },
            raw={"model": resp.model} if resp else {},
        )


class OpenAITarget(LLMTarget):
    """OpenAI API target (GPT-4, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: str | None = None,
        max_tokens: int = 1000,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        self._transport = transport
        if not self.api_key:
            raise ValueError("OpenAI API key required")

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    async def _complete(self, messages: list[Message]) -> TargetResponse:
        import time

        import httpx

        start = time.time()
        async with httpx.AsyncClient(transport=self._transport) as client:
            try:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": m.role, "content": m.content} for m in messages],
                        "max_tokens": self.max_tokens,
                    },
                    timeout=60.0,
                )
                if response.status_code == 429:
                    raise RateLimitError("OpenAI rate limit exceeded")
                response.raise_for_status()
                data = response.json()
                return TargetResponse(
                    content=data["choices"][0]["message"]["content"],
                    model=self.model,
                    tokens_used=data.get("usage", {}).get("total_tokens", 0),
                    latency_ms=(time.time() - start) * 1000,
                )
            except httpx.HTTPStatusError as e:
                raise TargetError(f"OpenAI API error: {e}") from e
            except httpx.RequestError as e:
                raise NetworkError(f"Network error: {e}") from e


class AnthropicTarget(LLMTarget):
    """Anthropic API target (Claude)."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 1000,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self._transport = transport
        if not self.api_key:
            raise ValueError("Anthropic API key required")

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    async def _complete(self, messages: list[Message]) -> TargetResponse:
        import time

        import httpx

        system = "\n".join(m.content for m in messages if m.role == "system")
        convo = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        payload: dict = {"model": self.model, "messages": convo, "max_tokens": self.max_tokens}
        if system:
            payload["system"] = system

        start = time.time()
        async with httpx.AsyncClient(transport=self._transport) as client:
            try:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=60.0,
                )
                if response.status_code == 429:
                    raise RateLimitError("Anthropic rate limit exceeded")
                response.raise_for_status()
                data = response.json()
                usage = data.get("usage", {})
                return TargetResponse(
                    content=data["content"][0]["text"],
                    model=self.model,
                    tokens_used=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                    latency_ms=(time.time() - start) * 1000,
                )
            except httpx.HTTPStatusError as e:
                raise TargetError(f"Anthropic API error: {e}") from e
            except httpx.RequestError as e:
                raise NetworkError(f"Network error: {e}") from e


class OllamaTarget(LLMTarget):
    """Local Ollama target (chat API)."""

    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self._transport = transport

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    async def _complete(self, messages: list[Message]) -> TargetResponse:
        import time

        import httpx

        start = time.time()
        async with httpx.AsyncClient(transport=self._transport) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [{"role": m.role, "content": m.content} for m in messages],
                        "stream": False,
                    },
                    timeout=120.0,
                )
                response.raise_for_status()
                data = response.json()
                return TargetResponse(
                    content=data["message"]["content"],
                    model=self.model,
                    tokens_used=data.get("eval_count", 0),
                    latency_ms=(time.time() - start) * 1000,
                )
            except httpx.HTTPStatusError as e:
                raise TargetError(f"Ollama API error: {e}") from e
            except httpx.RequestError as e:
                raise NetworkError(f"Network error: {e}") from e


class GeminiTarget(LLMTarget):
    """Google Gemini API target."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        max_tokens: int = 1000,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self.api_key = (
            api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        self.max_tokens = max_tokens
        self._transport = transport
        if not self.api_key:
            raise ValueError("Gemini API key required")

    @property
    def name(self) -> str:
        return f"gemini:{self.model}"

    async def _complete(self, messages: list[Message]) -> TargetResponse:
        import time

        import httpx

        system = "\n".join(m.content for m in messages if m.role == "system")
        contents = [
            {
                "role": "model" if m.role == "assistant" else "user",
                "parts": [{"text": m.content}],
            }
            for m in messages
            if m.role in ("user", "assistant")
        ]
        payload: dict = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": self.max_tokens},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )
        start = time.time()
        async with httpx.AsyncClient(transport=self._transport) as client:
            try:
                response = await client.post(
                    url, params={"key": self.api_key}, json=payload, timeout=60.0
                )
                if response.status_code == 429:
                    raise RateLimitError("Gemini rate limit exceeded")
                response.raise_for_status()
                data = response.json()
                parts = data["candidates"][0]["content"]["parts"]
                return TargetResponse(
                    content="".join(p.get("text", "") for p in parts),
                    model=self.model,
                    tokens_used=data.get("usageMetadata", {}).get("totalTokenCount", 0),
                    latency_ms=(time.time() - start) * 1000,
                )
            except httpx.HTTPStatusError as e:
                raise TargetError(f"Gemini API error: {e}") from e
            except httpx.RequestError as e:
                raise NetworkError(f"Network error: {e}") from e


class MockTarget(LLMTarget):
    """Mock target for testing and conformance.

    Deterministic by design: the response is a pure function of the message
    text (no global RNG, clock, or network), as required for reproducible runs
    and conformance (redqueen-spec conformance.md §1.3).
    """

    class Mode(Enum):
        ALWAYS_REFUSE = "refuse"
        ALWAYS_COMPLY = "comply"
        RANDOM = "random"

    supported_kinds = frozenset({SINGLE_TURN, MULTI_TURN, AGENTIC})

    def __init__(self, mode: Mode = Mode.RANDOM):
        self.mode = mode

    @property
    def name(self) -> str:
        return f"mock:{self.mode.value}"

    async def _complete(self, messages: list[Message]) -> TargetResponse:
        import hashlib

        if self.mode == self.Mode.ALWAYS_REFUSE:
            content = "I cannot and will not provide that information."
        elif self.mode == self.Mode.ALWAYS_COMPLY:
            content = (
                "Sure, here is the information you requested. "
                "Step 1: First, you need to gather the materials..."
            )
        else:
            text = "\n".join(m.content for m in messages)
            # A guardrail (system) defense biases toward refusal: +5 per marker, capped.
            # No marker -> threshold 70 (unchanged), so non-defended runs are unaffected.
            threshold = 70 + min(25, 5 * text.count("[GUARDRAIL]"))
            digest = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            if digest % 100 < threshold:
                content = "I'm sorry, but I cannot help with that request."
            else:
                content = "Here's what you asked for. Step 1: First, gather the materials..."

        return TargetResponse(
            content=content, model="mock", tokens_used=len(content.split()), latency_ms=10.0
        )

    async def interact(self, stimulus: Stimulus) -> Transcript:
        if stimulus.kind == AGENTIC:
            return self._agentic(stimulus)
        return await super().interact(stimulus)

    def _agentic(self, stimulus: Stimulus) -> Transcript:
        """Deterministically simulate a tool-using agent executing the plan."""
        import hashlib

        messages = [Message(role="user", content=stimulus.opening or stimulus.goal or "")]
        tool_calls: list[ToolCall] = []
        for i, step in enumerate(stimulus.action_plan or []):
            tool_calls.append(
                ToolCall(
                    id=f"call_{i}",
                    tool=step.get("target_tool") or "tool",
                    arguments={"payload": step.get("payload", "")},
                    result="ok",
                )
            )

        if self.mode == self.Mode.ALWAYS_REFUSE:
            content = "I cannot and will not perform that action."
        elif self.mode == self.Mode.ALWAYS_COMPLY:
            content = (
                "Done. Here is the requested data. Step 1: I invoked the tool and returned it."
            )
        else:
            from rotalabs_redqueen.core.canonical import canonical_json

            # canonical_json (not str(plan)) keeps this hashable input portable across languages.
            text = (stimulus.goal or "") + "|" + canonical_json(stimulus.action_plan or [])
            digest = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            if digest % 100 < 70:
                content = "I'm sorry, but I cannot complete that action."
            else:
                content = "Here is the result. Step 1: the tool returned the data."

        messages.append(Message(role="assistant", content=content, tool_calls=tool_calls or None))
        return Transcript(
            target_id=self.id,
            stimulus_kind=AGENTIC,
            messages=messages,
            tool_calls=tool_calls,
            stop_reason=STOP_COMPLETED,
            usage={"input_tokens": 0, "output_tokens": len(content.split()), "wall_ms": 10},
            raw={"model": "mock"},
        )


def create_target(target_spec: str) -> LLMTarget:
    """Create a target from a ``provider:model`` spec string."""
    parts = target_spec.split(":", 1)
    provider = parts[0].lower()
    model = parts[1] if len(parts) > 1 else None

    if provider == "openai":
        return OpenAITarget(model=model or "gpt-4")
    elif provider == "anthropic":
        return AnthropicTarget(model=model or "claude-sonnet-4-20250514")
    elif provider == "ollama":
        return OllamaTarget(model=model or "llama2")
    elif provider == "gemini":
        return GeminiTarget(model=model or "gemini-2.0-flash")
    elif provider == "mock":
        mode_map = {
            "refuse": MockTarget.Mode.ALWAYS_REFUSE,
            "comply": MockTarget.Mode.ALWAYS_COMPLY,
            "random": MockTarget.Mode.RANDOM,
        }
        return MockTarget(mode=mode_map.get(model or "random", MockTarget.Mode.RANDOM))
    else:
        raise ValueError(f"Unknown provider: {provider}")
