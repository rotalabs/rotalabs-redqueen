"""Stimulus, Transcript, and message types (redqueen-spec wire types).

The central generalization of the framework: a genome's phenotype is a
``Stimulus`` -- a single prompt, a multi-turn conversation, or an agentic
action plan -- not a bare string. A ``Target`` executes a ``Stimulus`` and
returns a ``Transcript``. This is what lets one engine span single-turn,
multi-turn, and agentic/MCP attack surfaces.

See `_dev/redqueen-spec/types.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SPEC_VERSION = "0.1.0"

# Stimulus kinds
SINGLE_TURN = "single_turn"
MULTI_TURN = "multi_turn"
AGENTIC = "agentic"

# Transcript stop reasons
STOP_COMPLETED = "completed"
STOP_MAX_TURNS = "max_turns"
STOP_MAX_STEPS = "max_steps"
STOP_REFUSED = "refused"
STOP_ERROR = "error"


@dataclass
class ToolCall:
    """A single tool/MCP-method invocation within an agentic interaction."""

    id: str
    tool: str
    arguments: dict = field(default_factory=dict)
    result: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tool": self.tool,
            "arguments": self.arguments,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class Message:
    """A conversation message."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_calls: list[ToolCall] | None = None
    name: str | None = None

    def to_dict(self) -> dict:
        d: dict = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.name:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, data: dict | Message) -> Message:
        if isinstance(data, Message):
            return data
        tcs = data.get("tool_calls")
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            tool_calls=[ToolCall(**tc) for tc in tcs] if tcs else None,
            name=data.get("name"),
        )


@dataclass
class Stimulus:
    """A tagged-union phenotype. Use the classmethod constructors.

    ``kind`` selects which fields are meaningful (see `types.md`).
    """

    kind: str
    # single_turn
    prompt: str | None = None
    system: str | None = None
    # multi_turn
    mode: str | None = None  # "scripted" | "adaptive"
    turns: list[Message] | None = None
    policy_ref: str | None = None
    max_turns: int = 8
    # agentic
    goal: str | None = None
    opening: str | None = None
    available_tools: list[str] | None = None
    action_plan: list[dict] | None = None
    max_steps: int = 12

    @classmethod
    def single_turn(cls, prompt: str, system: str | None = None) -> Stimulus:
        return cls(kind=SINGLE_TURN, prompt=prompt, system=system)

    @classmethod
    def multi_turn(
        cls,
        turns: list[Message | dict] | None = None,
        mode: str = "scripted",
        policy_ref: str | None = None,
        max_turns: int = 8,
    ) -> Stimulus:
        return cls(
            kind=MULTI_TURN,
            mode=mode,
            turns=[Message.from_dict(t) for t in (turns or [])],
            policy_ref=policy_ref,
            max_turns=max_turns,
        )

    @classmethod
    def agentic(
        cls,
        goal: str,
        opening: str,
        available_tools: list[str],
        action_plan: list[dict],
        max_steps: int = 12,
    ) -> Stimulus:
        return cls(
            kind=AGENTIC,
            goal=goal,
            opening=opening,
            available_tools=list(available_tools),
            action_plan=list(action_plan),
            max_steps=max_steps,
        )

    def to_dict(self) -> dict:
        if self.kind == SINGLE_TURN:
            d = {"kind": self.kind, "prompt": self.prompt}
            if self.system:
                d["system"] = self.system
            return d
        if self.kind == MULTI_TURN:
            return {
                "kind": self.kind,
                "mode": self.mode or "scripted",
                "turns": [m.to_dict() for m in (self.turns or [])],
                "policy_ref": self.policy_ref,
                "max_turns": self.max_turns,
            }
        if self.kind == AGENTIC:
            return {
                "kind": self.kind,
                "goal": self.goal,
                "opening": self.opening,
                "available_tools": self.available_tools or [],
                "action_plan": self.action_plan or [],
                "max_steps": self.max_steps,
            }
        return {"kind": self.kind}


@dataclass
class Transcript:
    """The replayable record of executing a Stimulus against a target."""

    target_id: str
    stimulus_kind: str
    messages: list[Message] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_state: dict = field(default_factory=dict)
    stop_reason: str = STOP_COMPLETED
    usage: dict = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "wall_ms": 0}
    )
    raw: dict = field(default_factory=dict)
    spec_version: str = SPEC_VERSION

    @property
    def assistant_text(self) -> str:
        """All assistant message content, joined -- what a judge scores."""
        return "\n".join(m.content for m in self.messages if m.role == "assistant")

    @property
    def last_reply(self) -> str:
        for m in reversed(self.messages):
            if m.role == "assistant":
                return m.content
        return ""

    def to_dict(self) -> dict:
        return {
            "spec_version": self.spec_version,
            "target_id": self.target_id,
            "stimulus_kind": self.stimulus_kind,
            "messages": [m.to_dict() for m in self.messages],
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "final_state": self.final_state,
            "stop_reason": self.stop_reason,
            "usage": self.usage,
        }
