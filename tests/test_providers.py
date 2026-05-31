"""Provider adapter tests.

Offline by default: each adapter is given an injected ``httpx.MockTransport`` so we
verify request construction, response parsing, and error mapping with no network or
keys (CI-safe). Fixtures in ``tests/fixtures/providers/`` are real captured responses.

The ``*_live`` tests hit the real APIs and are skipped unless the relevant key is in
the environment (e.g. after sourcing ``.secrets.env``).
"""

import json
import os
from pathlib import Path

import httpx
import pytest

from rotalabs_redqueen.llm import (
    AnthropicTarget,
    GeminiTarget,
    NetworkError,
    OllamaTarget,
    OpenAITarget,
    RateLimitError,
    TargetError,
)

FIXTURES = Path(__file__).parent / "fixtures" / "providers"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _ollama_model() -> str | None:
    """First small local Ollama model, or None if the server isn't reachable."""
    try:
        models = [
            m["name"]
            for m in httpx.get("http://localhost:11434/api/tags", timeout=2).json().get("models", [])
        ]
    except Exception:
        return None
    for pref in ("mistral:7b", "llama3.1:8b"):
        if pref in models:
            return pref
    small = [m for m in models if any(s in m for s in ("7b", "8b", "3b", "1b", "mini", "small"))]
    return small[0] if small else (models[0] if models else None)


def _transport(captured: list, status: int = 200, body: dict | None = None, raise_exc=None):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if raise_exc is not None:
            raise raise_exc
        return httpx.Response(status, json=body if body is not None else {})

    return httpx.MockTransport(handler)


OLLAMA_FIXTURE = {
    "model": "llama2",
    "message": {"role": "assistant", "content": "pong"},
    "done": True,
    "eval_count": 7,
}


# --- request construction + response parsing (offline) -----------------------


async def test_openai_request_and_parse():
    fixture = _load("openai_chat")
    cap: list = []
    target = OpenAITarget(model="gpt-4o-mini", api_key="test-key", transport=_transport(cap, 200, fixture))
    resp = await target.query("ping")

    req = cap[0]
    assert req.method == "POST"
    assert str(req.url) == "https://api.openai.com/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer test-key"
    sent = json.loads(req.content)
    assert sent["model"] == "gpt-4o-mini"
    assert sent["messages"] == [{"role": "user", "content": "ping"}]

    assert resp.content == fixture["choices"][0]["message"]["content"]
    assert resp.model == "gpt-4o-mini"
    assert resp.tokens_used == fixture["usage"]["total_tokens"]


async def test_anthropic_request_and_parse():
    fixture = _load("anthropic_messages")
    cap: list = []
    target = AnthropicTarget(model="claude-x", api_key="test-key", transport=_transport(cap, 200, fixture))
    resp = await target.query("ping")

    req = cap[0]
    assert str(req.url) == "https://api.anthropic.com/v1/messages"
    assert req.headers["x-api-key"] == "test-key"
    assert req.headers["anthropic-version"] == "2023-06-01"
    sent = json.loads(req.content)
    assert sent["model"] == "claude-x"
    assert sent["messages"] == [{"role": "user", "content": "ping"}]

    assert resp.content == fixture["content"][0]["text"]
    usage = fixture["usage"]
    assert resp.tokens_used == usage["input_tokens"] + usage["output_tokens"]


async def test_anthropic_extracts_system_message():
    cap: list = []
    target = AnthropicTarget(api_key="t", transport=_transport(cap, 200, _load("anthropic_messages")))
    from rotalabs_redqueen import Stimulus

    await target.interact(Stimulus.single_turn("hello", system="be safe"))
    sent = json.loads(cap[0].content)
    assert sent["system"] == "be safe"
    assert all(m["role"] != "system" for m in sent["messages"])


async def test_gemini_request_and_parse():
    fixture = _load("gemini_generate")
    cap: list = []
    target = GeminiTarget(model="gemini-2.0-flash", api_key="test-key", transport=_transport(cap, 200, fixture))
    resp = await target.query("ping")

    req = cap[0]
    assert req.url.path == "/v1beta/models/gemini-2.0-flash:generateContent"
    assert req.url.params["key"] == "test-key"
    sent = json.loads(req.content)
    assert sent["contents"][0]["parts"][0]["text"] == "ping"

    parts = fixture["candidates"][0]["content"]["parts"]
    assert resp.content == "".join(p.get("text", "") for p in parts)
    assert resp.tokens_used == fixture["usageMetadata"]["totalTokenCount"]


async def test_ollama_request_and_parse():
    cap: list = []
    target = OllamaTarget(model="llama2", transport=_transport(cap, 200, OLLAMA_FIXTURE))
    resp = await target.query("ping")

    req = cap[0]
    assert str(req.url) == "http://localhost:11434/api/chat"
    sent = json.loads(req.content)
    assert sent["model"] == "llama2"
    assert sent["stream"] is False
    assert resp.content == "pong"
    assert resp.tokens_used == 7


# --- error mapping (offline) --------------------------------------------------


async def test_rate_limit_maps_to_rate_limit_error():
    with pytest.raises(RateLimitError):
        await OpenAITarget(api_key="t", transport=_transport([], 429, {})).query("x")
    with pytest.raises(RateLimitError):
        await GeminiTarget(api_key="t", transport=_transport([], 429, {})).query("x")


async def test_http_error_maps_to_target_error():
    with pytest.raises(TargetError):
        await OpenAITarget(api_key="t", transport=_transport([], 500, {})).query("x")
    with pytest.raises(TargetError):
        await AnthropicTarget(api_key="t", transport=_transport([], 400, {})).query("x")


async def test_network_error_maps_to_network_error():
    boom = httpx.ConnectError("connection refused")
    with pytest.raises(NetworkError):
        await OpenAITarget(api_key="t", transport=_transport([], raise_exc=boom)).query("x")
    with pytest.raises(NetworkError):
        await OllamaTarget(transport=_transport([], raise_exc=boom)).query("x")


# --- live smokes (skipped without keys) --------------------------------------

_PROMPT = "Reply with the single word: pong"


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
async def test_openai_live():
    resp = await OpenAITarget(model="gpt-4o-mini").query(_PROMPT)
    assert resp.content.strip()


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY")
async def test_anthropic_live():
    resp = await AnthropicTarget(model="claude-haiku-4-5-20251001").query(_PROMPT)
    assert resp.content.strip()


@pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="no GEMINI_API_KEY",
)
async def test_gemini_live():
    resp = await GeminiTarget(model="gemini-2.0-flash").query(_PROMPT)
    assert resp.content.strip()


@pytest.mark.skipif(_ollama_model() is None, reason="Ollama not reachable / no models")
async def test_ollama_live():
    resp = await OllamaTarget(model=_ollama_model()).query(_PROMPT)
    assert resp.content.strip()
