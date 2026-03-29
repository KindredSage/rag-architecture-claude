"""
Schema Analyzer: Introspects ClickHouse for the actual schema of odin.ch_gm_trade_body.

This is the FIRST node in the trade agent graph. It fetches ground-truth schema
and sample rows BEFORE any LLM reasoning happens, so all downstream nodes work
with real column names and real data -- no guessing.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from agents.trade.state import TradeAgentState

logger = logging.getLogger(__name__)

TARGET_TABLE = "ch_gm_trade_body"
TARGET_DATABASE = "odin"


def _format_sample_rows_table(sample_rows: list[dict], max_cols: int = 12) -> str:
    """Format sample rows as a markdown table for LLM consumption."""
    if not sample_rows:
        return "(no sample data available)"

    cols = list(sample_rows[0].keys())[:max_cols]
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for row in sample_rows:
        vals = []
        for c in cols:
            v = str(row.get(c, ""))
            vals.append(v[:40])  # truncate long values
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, separator] + rows)


async def schema_analyzer(state: TradeAgentState, *, ch_service, cache_service=None) -> dict:
    """
    Fetch actual schema + sample rows from ClickHouse.

    Targets odin.ch_gm_trade_body as the primary table. Falls back to full
    schema discovery if the targeted fetch fails.
    """
    start = time.perf_counter()

    try:
        # Try cache first
        schema_data = None
        cache_key = f"{TARGET_DATABASE}.{TARGET_TABLE}"
        if cache_service:
            schema_data = await cache_service.get_schema(cache_key)

        if not schema_data:
            # Primary path: fetch targeted schema + sample rows
            schema_data = ch_service.get_targeted_schema_context(
                TARGET_TABLE, database=TARGET_DATABASE, sample_limit=5,
            )

            # If targeted fetch failed, fall back to full schema discovery
            if schema_data.get("error") or not schema_data.get("tables"):
                logger.warning(
                    "Targeted schema fetch failed (%s), falling back to full discovery",
                    schema_data.get("error", "no tables"),
                )
                schema_data = ch_service.get_full_schema_context(TARGET_DATABASE)

            if cache_service and schema_data.get("tables"):
                await cache_service.set_schema(cache_key, schema_data)

        all_tables = schema_data.get("tables", {})

        if not all_tables:
            duration = (time.perf_counter() - start) * 1000
            return {
                "schema_info": {
                    "database": TARGET_DATABASE,
                    "target_table": TARGET_TABLE,
                    "tables": {},
                    "error": schema_data.get("error", "No tables found"),
                },
                "execution_trace": [{
                    "node": "schema_analyzer",
                    "status": "warning",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": round(duration, 1),
                    "output_summary": f"No tables found in {TARGET_DATABASE}",
                }],
            }

        # Build schema text for downstream LLM consumption
        schema_text_parts = []
        all_sample_rows = []

        for tbl_name, tbl_info in all_tables.items():
            cols = tbl_info.get("columns", [])
            col_lines = []
            for c in cols:
                parts = [f"    - {c['name']}: {c['type']}"]
                if c.get("comment"):
                    parts.append(f"  -- {c['comment']}")
                if c.get("is_partition_key"):
                    parts.append(" [PARTITION KEY]")
                if c.get("is_sorting_key"):
                    parts.append(" [SORTING KEY]")
                col_lines.append("".join(parts))

            schema_text_parts.append(
                f"TABLE: {TARGET_DATABASE}.{tbl_name}\n"
                f"  Engine: {tbl_info.get('engine', 'N/A')}\n"
                f"  Partition Key: {tbl_info.get('partition_key', 'N/A')}\n"
                f"  Sorting Key: {tbl_info.get('sorting_key', 'N/A')}\n"
                f"  Rows: {tbl_info.get('total_rows', 'N/A'):,}\n"
                f"  Size: {tbl_info.get('readable_size', 'N/A')}\n"
                f"  Columns:\n" + "\n".join(col_lines)
            )

            # Collect sample rows
            sample = tbl_info.get("sample_rows", [])
            if sample:
                all_sample_rows.extend(sample)

        schema_text = "\n\n".join(schema_text_parts)
        sample_rows_text = _format_sample_rows_table(all_sample_rows)

        # Infer relationships between tables (if multiple matched)
        relationships = []
        table_names = list(all_tables.keys())
        for i, t1 in enumerate(table_names):
            cols1 = {c["name"].lower() for c in all_tables[t1].get("columns", [])}
            for t2 in table_names[i + 1:]:
                cols2 = {c["name"].lower() for c in all_tables[t2].get("columns", [])}
                common = cols1 & cols2 - {"id", "created_at", "updated_at", "timestamp"}
                if common:
                    relationships.append(
                        f"{t1} <-> {t2} via [{', '.join(sorted(common))}]"
                    )

        duration = (time.perf_counter() - start) * 1000

        schema_info = {
            "database": TARGET_DATABASE,
            "target_table": TARGET_TABLE,
            "tables": all_tables,
            "schema_text": schema_text,
            "sample_rows_text": sample_rows_text,
            "sample_rows": all_sample_rows,
            "relationships": relationships,
            "total_tables_in_db": len(all_tables),
            "matched_count": len(all_tables),
        }

        logger.info(
            "Schema analyzed: %d tables, %d sample rows, %d columns (%.0fms)",
            len(all_tables),
            len(all_sample_rows),
            sum(len(t.get("columns", [])) for t in all_tables.values()),
            duration,
        )

        return {
            "schema_info": schema_info,
            "execution_trace": [{
                "node": "schema_analyzer",
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
                "output_summary": (
                    f"Schema for {TARGET_DATABASE}.{TARGET_TABLE}: "
                    f"{sum(len(t.get('columns', [])) for t in all_tables.values())} cols, "
                    f"{len(all_sample_rows)} sample rows"
                ),
            }],
        }

    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        logger.error("Schema analyzer failed: %s", e)
        return {
            "schema_info": {
                "database": TARGET_DATABASE,
                "target_table": TARGET_TABLE,
                "tables": {},
                "error": str(e),
            },
            "execution_trace": [{
                "node": "schema_analyzer",
                "status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
                "error": str(e),
            }],
        }
