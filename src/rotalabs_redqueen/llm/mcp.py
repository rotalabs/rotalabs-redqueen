"""MCP target: red-team a tool-using agent over the Model Context Protocol.

`MCPTarget` spawns an MCP server (stdio transport, JSON-RPC 2.0), performs the
initialize handshake, lists its tools, and executes an agentic `Stimulus`'s
action plan by calling those tools. The tool outputs become the transcript a
judge scores -- so an exfiltration/goal-hijack attack against a real MCP server
is evaluated exactly like any other surface.
"""

from __future__ import annotations

import asyncio
import json

from rotalabs_redqueen._version import __version__
from rotalabs_redqueen.core.stimulus import (
    AGENTIC,
    STOP_COMPLETED,
    Message,
    Stimulus,
    ToolCall,
    Transcript,
)
from rotalabs_redqueen.llm.targets import LLMTarget, TargetError, TargetResponse


class MCPTarget(LLMTarget):
    """A target backed by an MCP server reached over stdio.

    Args:
        command: the server launch command, e.g. ``["python", "server.py"]`` or
            ``["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]``.
        name: optional display name.
        protocol_version: MCP protocol version to advertise.
    """

    supported_kinds = frozenset({AGENTIC})

    def __init__(
        self,
        command: list[str],
        name: str | None = None,
        protocol_version: str = "2024-11-05",
    ):
        self.command = command
        self.protocol_version = protocol_version
        self._name = name or f"mcp:{command[-1].split('/')[-1]}"
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self.tools: list[dict] = []

    @property
    def name(self) -> str:
        return self._name

    async def _complete(self, messages: list[Message]) -> TargetResponse:
        raise TargetError("MCPTarget only supports agentic stimuli")

    async def _send(self, obj: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write((json.dumps(obj) + "\n").encode())
        await self._proc.stdin.drain()

    async def _rpc(self, method: str, params: dict) -> dict:
        assert self._proc is not None and self._proc.stdout is not None
        self._next_id += 1
        await self._send({"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params})
        line = await self._proc.stdout.readline()
        if not line:
            raise TargetError(f"MCP server closed the connection during {method}")
        resp = json.loads(line.decode())
        if "error" in resp:
            raise TargetError(f"MCP error in {method}: {resp['error'].get('message')}")
        return resp.get("result", {})

    async def _connect(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await self._rpc(
            "initialize",
            {
                "protocolVersion": self.protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "rotalabs-redqueen", "version": __version__},
            },
        )
        await self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.tools = (await self._rpc("tools/list", {})).get("tools", [])

    async def interact(self, stimulus: Stimulus) -> Transcript:
        if stimulus.kind != AGENTIC:
            raise TargetError(f"{self.id} only supports agentic stimuli, got '{stimulus.kind}'")
        if self._proc is None:
            await self._connect()

        messages: list[Message] = [Message(role="user", content=stimulus.opening or stimulus.goal or "")]
        tool_calls: list[ToolCall] = []
        for i, step in enumerate(stimulus.action_plan or []):
            tool = step.get("target_tool") or (self.tools[0]["name"] if self.tools else "tool")
            args = step.get("arguments") or {"input": step.get("payload", "")}
            tc = ToolCall(id=f"call_{i}", tool=tool, arguments=args)
            try:
                result = await self._rpc("tools/call", {"name": tool, "arguments": args})
                tc.result = "".join(
                    c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"
                )
                messages.append(Message(role="tool", content=tc.result, name=tool))
            except TargetError as e:
                tc.error = str(e)
            tool_calls.append(tc)

        # The concatenated tool output is what a judge scans (e.g. for leaked secrets).
        summary = "\n".join(tc.result for tc in tool_calls if tc.result)
        messages.append(Message(role="assistant", content=summary))
        return Transcript(
            target_id=self.id,
            stimulus_kind=AGENTIC,
            messages=messages,
            tool_calls=tool_calls,
            stop_reason=STOP_COMPLETED,
            raw={"tools": [t.get("name") for t in self.tools]},
        )

    async def aclose(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                await self._proc.wait()
            except ProcessLookupError:
                pass
            self._proc = None
