"""LLM target adapters for adversarial testing.

Provides a unified interface for querying different LLM providers.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class TargetError(Exception):
    """Error from LLM target."""

    pass


class RateLimitError(TargetError):
    """Rate limit exceeded."""

    pass


class NetworkError(TargetError):
    """Network connectivity error."""

    pass


@dataclass
class TargetResponse:
    """Response from LLM target."""

    content: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0


class LLMTarget(ABC):
    """Abstract base class for LLM targets."""

    @abstractmethod
    async def query(self, prompt: str) -> TargetResponse:
        """Send a prompt to the LLM and get response.

        Args:
            prompt: The prompt to send

        Returns:
            Target response with content and metadata

        Raises:
            TargetError: If the query fails
            RateLimitError: If rate limited
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this target."""
        pass


class OpenAITarget(LLMTarget):
    """OpenAI API target (GPT-4, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: str | None = None,
        max_tokens: int = 1000,
    ):
        """Initialize OpenAI target.

        Args:
            model: Model name (gpt-4, gpt-3.5-turbo, etc.)
            api_key: API key (or uses OPENAI_API_KEY env var)
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError("OpenAI API key required")

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    async def query(self, prompt: str) -> TargetResponse:
        """Query OpenAI API."""
        import time

        import httpx

        start = time.time()

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": self.max_tokens,
                    },
                    timeout=60.0,
                )

                if response.status_code == 429:
                    raise RateLimitError("OpenAI rate limit exceeded")
                response.raise_for_status()

                data = response.json()
                latency = (time.time() - start) * 1000

                return TargetResponse(
                    content=data["choices"][0]["message"]["content"],
                    model=self.model,
                    tokens_used=data.get("usage", {}).get("total_tokens", 0),
                    latency_ms=latency,
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
    ):
        """Initialize Anthropic target.

        Args:
            model: Model name (claude-sonnet-4-20250514, etc.)
            api_key: API key (or uses ANTHROPIC_API_KEY env var)
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError("Anthropic API key required")

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    async def query(self, prompt: str) -> TargetResponse:
        """Query Anthropic API."""
        import time

        import httpx

        start = time.time()

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": self.max_tokens,
                    },
                    timeout=60.0,
                )

                if response.status_code == 429:
                    raise RateLimitError("Anthropic rate limit exceeded")
                response.raise_for_status()

                data = response.json()
                latency = (time.time() - start) * 1000

                return TargetResponse(
                    content=data["content"][0]["text"],
                    model=self.model,
                    tokens_used=data.get("usage", {}).get("input_tokens", 0)
                    + data.get("usage", {}).get("output_tokens", 0),
                    latency_ms=latency,
                )
            except httpx.HTTPStatusError as e:
                raise TargetError(f"Anthropic API error: {e}") from e
            except httpx.RequestError as e:
                raise NetworkError(f"Network error: {e}") from e


class OllamaTarget(LLMTarget):
    """Local Ollama target."""

    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
    ):
        """Initialize Ollama target.

        Args:
            model: Model name
            base_url: Ollama API URL
        """
        self.model = model
        self.base_url = base_url

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    async def query(self, prompt: str) -> TargetResponse:
        """Query Ollama API."""
        import time

        import httpx

        start = time.time()

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                    },
                    timeout=120.0,
                )
                response.raise_for_status()

                data = response.json()
                latency = (time.time() - start) * 1000

                return TargetResponse(
                    content=data["response"],
                    model=self.model,
                    tokens_used=data.get("eval_count", 0),
                    latency_ms=latency,
                )
            except httpx.HTTPStatusError as e:
                raise TargetError(f"Ollama API error: {e}") from e
            except httpx.RequestError as e:
                raise NetworkError(f"Network error: {e}") from e


class MockTarget(LLMTarget):
    """Mock target for testing."""

    class Mode(Enum):
        ALWAYS_REFUSE = "refuse"
        ALWAYS_COMPLY = "comply"
        RANDOM = "random"

    def __init__(self, mode: Mode = Mode.RANDOM):
        """Initialize mock target.

        Args:
            mode: How to respond to queries
        """
        self.mode = mode

    @property
    def name(self) -> str:
        return f"mock:{self.mode.value}"

    async def query(self, prompt: str) -> TargetResponse:
        """Return mock response."""
        import numpy as np

        if self.mode == self.Mode.ALWAYS_REFUSE:
            content = "I cannot and will not provide that information."
        elif self.mode == self.Mode.ALWAYS_COMPLY:
            content = "Sure, here is the information you requested. Step 1: First, you need to gather the materials..."
        else:
            # Random - 70% refuse, 30% comply
            if np.random.random() < 0.7:
                content = "I'm sorry, but I cannot help with that request."
            else:
                content = "Here's what you asked for. Step 1: First, gather the materials..."

        return TargetResponse(
            content=content,
            model="mock",
            tokens_used=len(content.split()),
            latency_ms=10.0,
        )


def create_target(target_spec: str) -> LLMTarget:
    """Create a target from a specification string.

    Args:
        target_spec: Format "provider:model" (e.g., "openai:gpt-4")

    Returns:
        Configured LLM target
    """
    parts = target_spec.split(":", 1)
    provider = parts[0].lower()
    model = parts[1] if len(parts) > 1 else None

    if provider == "openai":
        return OpenAITarget(model=model or "gpt-4")
    elif provider == "anthropic":
        return AnthropicTarget(model=model or "claude-sonnet-4-20250514")
    elif provider == "ollama":
        return OllamaTarget(model=model or "llama2")
    elif provider == "mock":
        mode_map = {
            "refuse": MockTarget.Mode.ALWAYS_REFUSE,
            "comply": MockTarget.Mode.ALWAYS_COMPLY,
            "random": MockTarget.Mode.RANDOM,
        }
        mode = mode_map.get(model or "random", MockTarget.Mode.RANDOM)
        return MockTarget(mode=mode)
    else:
        raise ValueError(f"Unknown provider: {provider}")
