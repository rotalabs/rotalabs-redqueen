# MCP target

Red-team a tool-using agent over the Model Context Protocol (stdio JSON-RPC). `MCPTarget`
spawns an MCP server, performs the initialize handshake, lists its tools, and executes an
agentic `Stimulus`'s action plan as `tools/call` invocations — surfacing tool output for judging.

::: rotalabs_redqueen.llm.mcp.MCPTarget
