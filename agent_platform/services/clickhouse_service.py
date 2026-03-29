"""
ClickHouse connection service with safety guardrails.

Provides:
- Connection management with clickhouse-connect
- Read-only query execution with timeout & row limits
- Schema introspection helpers
- Programmatic SQL safety validation
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import clickhouse_connect
from clickhouse_connect.driver.client import Client

from config import Settings

logger = logging.getLogger(__name__)


class ClickHouseService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Client | None = None

    # ── Lifecycle ────────────────────────────────────────────────

    def initialize(self) -> None:
        self._client = clickhouse_connect.get_client(
            host=self.settings.ch_host,
            port=self.settings.ch_port,
            database=self.settings.ch_database,
            username=self.settings.ch_user,
            password=self.settings.ch_password.get_secret_value(),
            secure=self.settings.ch_secure,
            connect_timeout=self.settings.ch_connect_timeout,
            send_receive_timeout=self.settings.ch_send_receive_timeout,
            settings={
                "readonly": "1",
                "max_execution_time": str(self.settings.ch_query_timeout),
                "max_result_rows": str(self.settings.ch_max_rows),
            },
        )
        logger.info(
            "ClickHouse connected: %s:%s/%s",
            self.settings.ch_host,
            self.settings.ch_port,
            self.settings.ch_database,
        )

    def close(self) -> None:
        if self._client:
            self._client.close()
            logger.info("ClickHouse connection closed")

    @property
    def client(self) -> Client:
        if self._client is None:
            raise RuntimeError("ClickHouseService not initialized")
        return self._client

    def ping(self) -> bool:
        try:
            self.client.ping()
            return True
        except Exception:
            return False

    # ── SQL Safety ───────────────────────────────────────────────

    def validate_sql_safety(self, sql: str) -> tuple[bool, list[str]]:
        """
        Programmatic SQL safety check. Returns (is_safe, issues).
        This is defense-in-depth ON TOP of the readonly session setting.
        """
        issues: list[str] = []
        sql_upper = sql.upper().strip()

        # Block dangerous keywords
        for kw in self.settings.sql_blocked_keywords:
            pattern = rf"\b{kw}\b"
            if re.search(pattern, sql_upper):
                issues.append(f"Blocked keyword detected: {kw}")

        # Must start with SELECT, WITH, SHOW, DESCRIBE, or EXPLAIN
        allowed_starts = ("SELECT", "WITH", "SHOW", "DESCRIBE", "DESC", "EXPLAIN")
        if not any(sql_upper.startswith(s) for s in allowed_starts):
            issues.append(
                f"Query must start with one of: {', '.join(allowed_starts)}"
            )

        # Block semicolons (multi-statement injection)
        stripped = re.sub(r"'[^']*'", "", sql)  # remove string literals
        if ";" in stripped:
            issues.append("Semicolons not allowed (multi-statement risk)")

        # Block system database access unless explicitly allowed
        system_pattern = r"\bsystem\s*\.\s*(?!tables|columns|parts|databases|query_log)"
        if re.search(system_pattern, sql_upper):
            issues.append("Access to system tables is restricted")

        # Ensure LIMIT is present (performance safety)
        if "LIMIT" not in sql_upper and sql_upper.startswith("SELECT"):
            issues.append("SELECT queries must include a LIMIT clause")

        return len(issues) == 0, issues

    # ── Query Execution ──────────────────────────────────────────

    def execute_query(
        self,
        sql: str,
        parameters: dict[str, Any] | None = None,
        max_rows: int | None = None,
    ) -> dict[str, Any]:
        """
        Execute a read-only query and return structured results.
        """
        is_safe, issues = self.validate_sql_safety(sql)
        if not is_safe:
            return {
                "success": False,
                "data": [],
                "columns": [],
                "row_count": 0,
                "execution_time_ms": 0,
                "bytes_read": 0,
                "truncated": False,
                "error": f"SQL safety check failed: {'; '.join(issues)}",
            }

        effective_max = max_rows or self.settings.ch_max_rows
        start = time.perf_counter()

        try:
            result = self.client.query(
                sql,
                parameters=parameters or {},
                settings={
                    "max_result_rows": str(effective_max),
                    "max_execution_time": str(self.settings.ch_query_timeout),
                },
            )

            elapsed_ms = (time.perf_counter() - start) * 1000
            col_names = result.column_names
            col_types = [str(t) for t in result.column_types]

            rows = []
            for row in result.result_rows:
                row_dict = {}
                for i, val in enumerate(row):
                    # Handle special ClickHouse types
                    if hasattr(val, "isoformat"):
                        row_dict[col_names[i]] = val.isoformat()
                    elif isinstance(val, bytes):
                        row_dict[col_names[i]] = val.decode("utf-8", errors="replace")
                    else:
                        row_dict[col_names[i]] = val
                rows.append(row_dict)

            truncated = len(rows) >= effective_max

            logger.info(
                "Query executed: %d rows, %.1fms, %s",
                len(rows),
                elapsed_ms,
                sql[:100],
            )

            return {
                "success": True,
                "data": rows,
                "columns": [
                    {"name": n, "type": t}
                    for n, t in zip(col_names, col_types)
                ],
                "row_count": len(rows),
                "execution_time_ms": round(elapsed_ms, 2),
                "bytes_read": result.summary.get("read_bytes", 0) if result.summary else 0,
                "truncated": truncated,
                "error": None,
            }

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error("Query failed (%.1fms): %s | SQL: %s", elapsed_ms, e, sql[:200])
            return {
                "success": False,
                "data": [],
                "columns": [],
                "row_count": 0,
                "execution_time_ms": round(elapsed_ms, 2),
                "bytes_read": 0,
                "truncated": False,
                "error": str(e),
            }

    # ── Schema Introspection ─────────────────────────────────────

    def get_tables(self, database: str | None = None) -> list[dict[str, Any]]:
        db = database or self.settings.ch_database
        if db not in self.settings.allowed_ch_databases:
            return []

        result = self.execute_query(
            """
            SELECT
                database, name, engine, partition_key, sorting_key,
                total_rows, total_bytes,
                formatReadableSize(total_bytes) as readable_size
            FROM system.tables
            WHERE database = {db:String}
            ORDER BY name
            LIMIT 200
            """,
            parameters={"db": db},
        )
        return result["data"] if result["success"] else []

    def get_columns(
        self,
        table_name: str,
        database: str | None = None,
    ) -> list[dict[str, Any]]:
        db = database or self.settings.ch_database
        if db not in self.settings.allowed_ch_databases:
            return []

        result = self.execute_query(
            """
            SELECT name, type, comment, is_in_partition_key, is_in_sorting_key
            FROM system.columns
            WHERE database = {db:String} AND table = {tbl:String}
            ORDER BY position
            LIMIT 500
            """,
            parameters={"db": db, "tbl": table_name},
        )
        return result["data"] if result["success"] else []

    def get_table_sample(
        self,
        table_name: str,
        limit: int = 5,
        database: str | None = None,
    ) -> list[dict[str, Any]]:
        db = database or self.settings.ch_database
        if db not in self.settings.allowed_ch_databases:
            return []

        safe_limit = min(limit, 20)
        result = self.execute_query(
            f"SELECT * FROM {db}.{table_name} LIMIT {safe_limit}",
        )
        return result["data"] if result["success"] else []

    def get_targeted_schema_context(
        self,
        table_name: str,
        database: str | None = None,
        sample_limit: int = 5,
    ) -> dict[str, Any]:
        """
        Fetch schema + sample rows for a single specific table.
        Used by trade agent to get ground-truth schema before any LLM reasoning.
        """
        db = database or self.settings.ch_database
        if db not in self.settings.allowed_ch_databases:
            return {"database": db, "tables": {}, "error": f"Database {db} not in allowed list"}

        tables_result = self.execute_query(
            """
            SELECT
                database, name, engine, partition_key, sorting_key,
                total_rows, total_bytes,
                formatReadableSize(total_bytes) as readable_size
            FROM system.tables
            WHERE database = {db:String} AND name = {tbl:String}
            LIMIT 1
            """,
            parameters={"db": db, "tbl": table_name},
        )

        if not tables_result["success"] or not tables_result["data"]:
            return {"database": db, "tables": {}, "error": f"Table {db}.{table_name} not found"}

        tbl_meta = tables_result["data"][0]
        columns = self.get_columns(table_name, db)
        sample_rows = self.get_table_sample(table_name, limit=sample_limit, database=db)

        schema = {
            table_name: {
                "engine": tbl_meta.get("engine", ""),
                "partition_key": tbl_meta.get("partition_key", ""),
                "sorting_key": tbl_meta.get("sorting_key", ""),
                "total_rows": tbl_meta.get("total_rows", 0),
                "readable_size": tbl_meta.get("readable_size", ""),
                "columns": [
                    {
                        "name": c["name"],
                        "type": c["type"],
                        "comment": c.get("comment", ""),
                        "is_partition_key": c.get("is_in_partition_key", 0) == 1,
                        "is_sorting_key": c.get("is_in_sorting_key", 0) == 1,
                    }
                    for c in columns
                ],
                "sample_rows": sample_rows,
            }
        }

        return {"database": db, "tables": schema}

    def explain_query(self, sql: str) -> str:
        result = self.execute_query(f"EXPLAIN {sql}")
        if result["success"] and result["data"]:
            return "\n".join(
                str(row.get("explain", row)) for row in result["data"]
            )
        return result.get("error", "EXPLAIN failed")

    def get_full_schema_context(self, database: str | None = None) -> dict[str, Any]:
        """Build a comprehensive schema context for LLM consumption."""
        db = database or self.settings.ch_database
        tables = self.get_tables(db)
        schema: dict[str, Any] = {}

        for tbl in tables:
            tbl_name = tbl["name"]
            columns = self.get_columns(tbl_name, db)
            schema[tbl_name] = {
                "engine": tbl.get("engine", ""),
                "partition_key": tbl.get("partition_key", ""),
                "sorting_key": tbl.get("sorting_key", ""),
                "total_rows": tbl.get("total_rows", 0),
                "readable_size": tbl.get("readable_size", ""),
                "columns": [
                    {
                        "name": c["name"],
                        "type": c["type"],
                        "comment": c.get("comment", ""),
                        "is_partition_key": c.get("is_in_partition_key", 0) == 1,
                        "is_sorting_key": c.get("is_in_sorting_key", 0) == 1,
                    }
                    for c in columns
                ],
            }

        return {"database": db, "tables": schema}
