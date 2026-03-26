"""
Schema Analyzer: Introspects ClickHouse system tables for actual schema info.

This is one of the few nodes that makes REAL database calls (read-only system queries).
It uses the ClickHouseService or MCP tools depending on configuration.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from agents.trade.state import TradeAgentState

logger = logging.getLogger(__name__)


async def schema_analyzer(state: TradeAgentState, *, ch_service, cache_service=None) -> dict:
    """
    Fetch actual schema from ClickHouse and match against suggested tables.

    This node does NOT use an LLM -- it queries ClickHouse system tables directly
    for ground-truth schema information.
    """
    start = time.perf_counter()

    trade_context = state.get("trade_context", {})
    suggested_tables = trade_context.get("suggested_tables", [])
    parsed_intent = state.get("parsed_intent", {})
    target_entities = parsed_intent.get("target_entities", [])

    # Combine all table hints
    candidate_tables = list(set(suggested_tables + target_entities))

    try:
        # Try cache first
        schema_data = None
        if cache_service:
            schema_data = await cache_service.get_schema(ch_service.settings.ch_database)

        if not schema_data:
            schema_data = ch_service.get_full_schema_context()
            if cache_service and schema_data.get("tables"):
                await cache_service.set_schema(ch_service.settings.ch_database, schema_data)

        all_tables = schema_data.get("tables", {})

        if not all_tables:
            duration = (time.perf_counter() - start) * 1000
            return {
                "schema_info": {"tables": {}, "error": "No tables found in database"},
                "execution_trace": [{
                    "node": "schema_analyzer",
                    "status": "warning",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": round(duration, 1),
                    "output_summary": "No tables found",
                }],
            }

        # Match candidates to actual tables
        matched_tables = {}
        unmatched = []

        for candidate in candidate_tables:
            candidate_lower = candidate.lower().strip()
            # Exact match
            if candidate_lower in {k.lower() for k in all_tables}:
                for real_name, info in all_tables.items():
                    if real_name.lower() == candidate_lower:
                        matched_tables[real_name] = info
                        break
            else:
                # Fuzzy match: check if candidate is a substring
                for real_name, info in all_tables.items():
                    if candidate_lower in real_name.lower() or real_name.lower() in candidate_lower:
                        matched_tables[real_name] = info
                        break
                else:
                    unmatched.append(candidate)

        # If no candidates matched, include ALL tables as context
        if not matched_tables:
            matched_tables = all_tables

        # Build schema context string for downstream LLM consumption
        schema_text_parts = []
        for tbl_name, tbl_info in matched_tables.items():
            cols = tbl_info.get("columns", [])
            col_lines = []
            for c in cols:
                parts = [f"  - {c['name']}: {c['type']}"]
                if c.get("comment"):
                    parts.append(f"  -- {c['comment']}")
                if c.get("is_partition_key"):
                    parts.append(" [PARTITION KEY]")
                if c.get("is_sorting_key"):
                    parts.append(" [SORTING KEY]")
                col_lines.append("".join(parts))

            schema_text_parts.append(
                f"TABLE: {tbl_name}\n"
                f"  Engine: {tbl_info.get('engine', 'N/A')}\n"
                f"  Partition Key: {tbl_info.get('partition_key', 'N/A')}\n"
                f"  Sorting Key: {tbl_info.get('sorting_key', 'N/A')}\n"
                f"  Rows: {tbl_info.get('total_rows', 'N/A'):,}\n"
                f"  Size: {tbl_info.get('readable_size', 'N/A')}\n"
                f"  Columns:\n" + "\n".join(col_lines)
            )

        schema_text = "\n\n".join(schema_text_parts)

        # Infer relationships between matched tables (basic FK detection)
        relationships = []
        table_names = list(matched_tables.keys())
        for i, t1 in enumerate(table_names):
            cols1 = {c["name"].lower() for c in matched_tables[t1].get("columns", [])}
            for t2 in table_names[i + 1:]:
                cols2 = {c["name"].lower() for c in matched_tables[t2].get("columns", [])}
                common = cols1 & cols2
                # Filter out generic columns
                common -= {"id", "created_at", "updated_at", "timestamp"}
                if common:
                    relationships.append(
                        f"{t1} <-> {t2} via [{', '.join(sorted(common))}]"
                    )

        duration = (time.perf_counter() - start) * 1000

        schema_info = {
            "database": schema_data.get("database", ""),
            "tables": matched_tables,
            "schema_text": schema_text,
            "relationships": relationships,
            "unmatched_candidates": unmatched,
            "total_tables_in_db": len(all_tables),
            "matched_count": len(matched_tables),
        }

        logger.info(
            "Schema analyzed: %d/%d tables matched, %d relationships (%.0fms)",
            len(matched_tables),
            len(all_tables),
            len(relationships),
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
                    f"Matched {len(matched_tables)} tables: {list(matched_tables.keys())[:5]}"
                ),
            }],
        }

    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        logger.error("Schema analyzer failed: %s", e)
        return {
            "schema_info": {"tables": {}, "error": str(e)},
            "execution_trace": [{
                "node": "schema_analyzer",
                "status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
                "error": str(e),
            }],
        }
