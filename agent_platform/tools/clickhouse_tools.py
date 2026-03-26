"""
LangChain-compatible tools for ClickHouse operations.

These tools are bound to agent nodes via bind_tools() or ToolNode.
Each tool receives the ClickHouseService instance via closure at registration time.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from services.clickhouse_service import ClickHouseService

# ---------------------------------------------------------------------------
# Tool factory: called once at startup with the live service instance.
# Returns a list of LangChain tools with the service captured in closure.
# ---------------------------------------------------------------------------


def create_clickhouse_tools(ch: ClickHouseService) -> list:
    """Build all ClickHouse tools with the service injected."""

    @tool
    def execute_clickhouse_query(
        sql: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a read-only SQL query against ClickHouse.
        Returns rows as a list of dicts plus metadata.
        ONLY SELECT / WITH / SHOW / DESCRIBE / EXPLAIN queries are allowed.
        A LIMIT clause is required for SELECT queries.
        Max 50,000 rows. Timeout 30s."""
        return ch.execute_query(sql, parameters)

    @tool
    def get_table_schema(table_name: str) -> list[dict[str, Any]]:
        """Get column names, types, comments, and key info for a ClickHouse table."""
        return ch.get_columns(table_name)

    @tool
    def list_clickhouse_tables() -> list[dict[str, Any]]:
        """List all tables in the configured ClickHouse database with
        engine type, partition key, sorting key, row count, and size."""
        return ch.get_tables()

    @tool
    def get_table_sample(table_name: str, limit: int = 5) -> list[dict[str, Any]]:
        """Return a small sample of rows from a ClickHouse table.
        Useful for understanding the data shape. Max 20 rows."""
        return ch.get_table_sample(table_name, min(limit, 20))

    @tool
    def explain_clickhouse_query(sql: str) -> str:
        """Run EXPLAIN on a ClickHouse query to show the execution plan.
        Useful for performance analysis."""
        return ch.explain_query(sql)

    @tool
    def validate_sql_safety(sql: str) -> dict[str, Any]:
        """Check if a SQL query passes safety validation.
        Returns {'is_safe': bool, 'issues': [...]}."""
        is_safe, issues = ch.validate_sql_safety(sql)
        return {"is_safe": is_safe, "issues": issues}

    return [
        execute_clickhouse_query,
        get_table_schema,
        list_clickhouse_tables,
        get_table_sample,
        explain_clickhouse_query,
        validate_sql_safety,
    ]
