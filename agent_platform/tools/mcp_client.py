"""
MCP (Model Context Protocol) client integration.

When MCP is enabled, this module connects to configured MCP tool servers
and exposes their tools as LangChain-compatible tools that can be injected
into agent graphs identically to native tools.

This makes the agent graph code agnostic to whether tools are native or MCP-based.
"""

from __future__ import annotations

import logging
from typing import Any

from config import Settings

logger = logging.getLogger(__name__)


class MCPClientManager:
    """Manages connections to one or more MCP servers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._tools: list = []
        self._client: Any = None

    async def initialize(self) -> None:
        """Connect to all configured MCP servers and collect tools."""
        if not self.settings.mcp_enabled or not self.settings.mcp_server_urls:
            logger.info("MCP disabled or no servers configured")
            return

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            server_config = {}
            for i, url in enumerate(self.settings.mcp_server_urls):
                server_name = f"mcp_server_{i}"
                # Auto-detect transport based on URL
                if url.endswith("/sse"):
                    transport = "sse"
                else:
                    transport = "streamable_http"

                server_config[server_name] = {
                    "url": url,
                    "transport": transport,
                }

            self._client = MultiServerMCPClient(server_config)
            self._tools = await self._client.get_tools()

            tool_names = [t.name for t in self._tools]
            logger.info(
                "MCP initialized: %d servers, %d tools: %s",
                len(self.settings.mcp_server_urls),
                len(self._tools),
                tool_names,
            )

        except ImportError:
            logger.error(
                "langchain-mcp-adapters not installed. "
                "Install with: pip install langchain-mcp-adapters"
            )
        except Exception as e:
            logger.error("MCP initialization failed: %s", e)

    async def shutdown(self) -> None:
        if self._client and hasattr(self._client, "close"):
            await self._client.close()

    @property
    def tools(self) -> list:
        return self._tools

    @property
    def available(self) -> bool:
        return len(self._tools) > 0

    def get_tools_by_prefix(self, prefix: str) -> list:
        """Get MCP tools whose names start with a prefix (e.g., 'clickhouse_')."""
        return [t for t in self._tools if t.name.startswith(prefix)]
