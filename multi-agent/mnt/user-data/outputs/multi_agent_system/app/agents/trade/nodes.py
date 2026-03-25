"""
agents/trade/nodes.py
---------------------
Eight sub-agent nodes for the Trade Agent pipeline.
Each node has a single responsibility, accepts TradeState,
and returns only the fields it mutated.

Sub-agents:
  1. QueryAnalyst        – parse query, extract entities
  2. QueryPlanner        – decide execution steps
  3. SchemaAnalyzer      – fetch / simulate schema for the target table
  4. QueryBuilder        – build the SQL query
  5. QueryValidator      – validate SQL (syntax + logic)
  6. QueryExecutor       – run the query (simulated DB layer)
  7. DetailsAnalyzer     – interpret results, flag anomalies
  8. ResultSynthesizer   – produce the final human-readable answer
"""

from __future__ import annotations
import json
import re
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.agents.trade.state import TradeState
from app.core.config import get_settings
from app.core.logging import get_logger
from app.utils.retry import with_retry

logger = get_logger(__name__)
settings = get_settings()


def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )


def _step(node: str, action: str, output: Any) -> dict:
    return {"node": node, "action": action, "output": output}


def _parse_json_response(text: str) -> dict:
    """Strip markdown fences and parse JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    return json.loads(text)


# ─────────────────────────────────────────────────────────────────────────────
# 1. QUERY ANALYST – understand what the user wants
# ─────────────────────────────────────────────────────────────────────────────

@with_retry(max_attempts=3, exceptions=(Exception,))
async def query_analyst(state: TradeState) -> dict:
    node_name = "QueryAnalyst"
    logger.info("node_start", node=node_name)

    system = SystemMessage(content=(
        "You are a query analysis expert for a trade data system. "
        "Extract structured information from the user's query.\n\n"
        "Respond ONLY with JSON:\n"
        "{\n"
        '  "intent": "retrieve | filter | aggregate | compare | other",\n'
        '  "table_name": "<table name or null>",\n'
        '  "file_id": "<id value or null>",\n'
        '  "filters": {"<col>": "<val>"},\n'
        '  "columns_requested": ["col1", ...] or ["*"],\n'
        '  "requires_join": false,\n'
        '  "summary": "<one sentence description of what the user wants>"\n'
        "}"
    ))
    human = HumanMessage(content=f"Query: {state['query']}")
    response = await _llm().ainvoke([system, human])

    analysis = _parse_json_response(response.content)
    logger.info("query_analyzed", node=node_name, intent=analysis.get("intent"))

    return {
        "query_analysis": analysis,
        "table_name": analysis.get("table_name"),
        "file_id": str(analysis.get("file_id")) if analysis.get("file_id") else None,
        "filters": analysis.get("filters", {}),
        "steps": [_step(node_name, "query_analyzed", analysis)],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. QUERY PLANNER – decide execution strategy
# ─────────────────────────────────────────────────────────────────────────────

@with_retry(max_attempts=3, exceptions=(Exception,))
async def query_planner(state: TradeState) -> dict:
    node_name = "QueryPlanner"
    logger.info("node_start", node=node_name)

    analysis_str = json.dumps(state["query_analysis"])
    system = SystemMessage(content=(
        "You are an execution planner for a data retrieval system. "
        "Given a query analysis, produce an ordered list of steps to fulfill the request.\n"
        "Respond ONLY with JSON: {\"plan\": [\"step1\", \"step2\", ...]}"
    ))
    human = HumanMessage(content=f"Query analysis: {analysis_str}")
    response = await _llm().ainvoke([system, human])

    parsed = _parse_json_response(response.content)
    plan = parsed.get("plan", ["Retrieve data", "Format response"])

    return {
        "execution_plan": plan,
        "steps": [_step(node_name, "plan_created", plan)],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. SCHEMA ANALYZER – get table schema (simulated)
# ─────────────────────────────────────────────────────────────────────────────

async def schema_analyzer(state: TradeState) -> dict:
    """
    In production this would hit your metadata store / information_schema.
    Here we simulate a schema catalogue for demo purposes.
    """
    node_name = "SchemaAnalyzer"
    logger.info("node_start", node=node_name, table=state.get("table_name"))

    # ── Simulated schema catalogue ────────────────────────────────────────────
    catalogue: dict[str, dict] = {
        "My_Table": {
            "columns": {
                "file_id": {"type": "BIGINT", "nullable": False, "primary_key": True},
                "trade_id": {"type": "VARCHAR(64)", "nullable": False, "indexed": True},
                "trade_date": {"type": "DATE", "nullable": False},
                "instrument": {"type": "VARCHAR(128)", "nullable": True},
                "quantity": {"type": "DECIMAL(18,6)", "nullable": True},
                "price": {"type": "DECIMAL(18,6)", "nullable": True},
                "notional": {"type": "DECIMAL(24,6)", "nullable": True},
                "currency": {"type": "CHAR(3)", "nullable": True},
                "counterparty": {"type": "VARCHAR(256)", "nullable": True},
                "status": {"type": "VARCHAR(32)", "nullable": True},
                "created_at": {"type": "TIMESTAMP", "nullable": False},
                "updated_at": {"type": "TIMESTAMP", "nullable": True},
            },
            "indexes": ["file_id", "trade_id", "trade_date"],
            "row_count_estimate": 4_200_000,
        }
    }

    table = state.get("table_name") or "unknown"
    schema = catalogue.get(table, {
        "columns": {},
        "indexes": [],
        "note": f"Table '{table}' not found in catalogue; proceeding with best-effort query.",
    })

    return {
        "schema_info": schema,
        "steps": [_step(node_name, "schema_fetched", {"table": table, "column_count": len(schema.get("columns", {}))})],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. QUERY BUILDER – build the SQL
# ─────────────────────────────────────────────────────────────────────────────

@with_retry(max_attempts=3, exceptions=(Exception,))
async def query_builder(state: TradeState) -> dict:
    node_name = "QueryBuilder"
    logger.info("node_start", node=node_name)

    schema_str = json.dumps(state.get("schema_info", {}), indent=2)
    analysis_str = json.dumps(state.get("query_analysis", {}), indent=2)

    system = SystemMessage(content=(
        "You are a SQL query builder for a trade data warehouse. "
        "Given the table schema and query analysis, produce a safe SELECT query.\n\n"
        "Rules:\n"
        "  • Use parameterised-style placeholders shown literally (e.g. WHERE file_id = :file_id).\n"
        "  • Never use DELETE / UPDATE / INSERT / DROP.\n"
        "  • Limit results to 1000 rows unless aggregation is requested.\n"
        "  • Use ANSI SQL.\n\n"
        "Respond ONLY with JSON: {\"sql\": \"<your SQL>\", \"params\": {\"param\": value}}"
    ))
    human = HumanMessage(content=(
        f"Schema:\n{schema_str}\n\n"
        f"Query analysis:\n{analysis_str}\n\n"
        f"Additional context: {json.dumps(state.get('context', {}))}"
    ))

    response = await _llm().ainvoke([system, human])
    parsed = _parse_json_response(response.content)
    sql = parsed.get("sql", "SELECT * FROM unknown LIMIT 10")
    params = parsed.get("params", {})

    # Inject known values from state if LLM left them as placeholders
    if state.get("file_id"):
        params.setdefault("file_id", state["file_id"])
    if state.get("table_name"):
        sql = sql.replace(":table_name", state["table_name"])

    return {
        "raw_query": sql,
        "steps": [_step(node_name, "query_built", {"sql": sql, "params": params})],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. QUERY VALIDATOR – sanity-check the SQL
# ─────────────────────────────────────────────────────────────────────────────

async def query_validator(state: TradeState) -> dict:
    """
    Lightweight rule-based + LLM validation.
    Catches: dangerous keywords, missing WHERE on large tables, syntax issues.
    """
    node_name = "QueryValidator"
    logger.info("node_start", node=node_name)

    sql = state.get("raw_query", "")
    errors: list[str] = []

    # ── Rule-based checks ─────────────────────────────────────────────────────
    dangerous = re.compile(r"\b(DELETE|UPDATE|INSERT|DROP|TRUNCATE|ALTER|EXEC|EXECUTE)\b", re.IGNORECASE)
    if dangerous.search(sql):
        errors.append("Query contains a dangerous DML/DDL keyword and was rejected.")

    if "WHERE" not in sql.upper() and "LIMIT" not in sql.upper():
        errors.append("Query has no WHERE clause or LIMIT – added LIMIT 100 as a safety guard.")
        sql += " LIMIT 100"

    # ── LLM structural check ──────────────────────────────────────────────────
    if not errors:
        system = SystemMessage(content=(
            "You are a SQL reviewer. Check this SQL for syntax issues or logical errors. "
            "Respond ONLY with JSON: "
            "{\"valid\": true/false, \"issues\": [\"issue1\", ...], \"corrected_sql\": \"<sql or empty>\"}"
        ))
        human = HumanMessage(content=f"SQL:\n{sql}")
        response = await _llm().ainvoke([system, human])
        review = _parse_json_response(response.content)
        if not review.get("valid", True):
            errors.extend(review.get("issues", []))
        if review.get("corrected_sql"):
            sql = review["corrected_sql"]

    validated_sql = sql if not any("rejected" in e for e in errors) else ""

    return {
        "validated_query": validated_sql,
        "validation_errors": errors,
        "steps": [_step(node_name, "query_validated", {"errors": errors, "passed": len(errors) == 0})],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. QUERY EXECUTOR – run the query (simulated DB)
# ─────────────────────────────────────────────────────────────────────────────

async def query_executor(state: TradeState) -> dict:
    """
    In production: connects to ClickHouse / Oracle / PG and runs the query.
    Here we return a realistic simulated result based on the extracted entities.
    """
    node_name = "QueryExecutor"
    logger.info("node_start", node=node_name)

    sql = state.get("validated_query", "")
    file_id = state.get("file_id", "999")

    if not sql:
        return {
            "query_result": {"rows": [], "error": "No valid query to execute."},
            "row_count": 0,
            "steps": [_step(node_name, "execution_skipped", "no valid query")],
            "error": "Query validation failed; execution skipped.",
        }

    # ── Simulated result set ───────────────────────────────────────────────────
    mock_result = {
        "rows": [
            {
                "file_id": int(file_id) if file_id else 999,
                "trade_id": "TRD-20241115-0042",
                "trade_date": "2024-11-15",
                "instrument": "USD/EUR FX Forward",
                "quantity": 1_000_000.00,
                "price": 1.0852,
                "notional": 1_085_200.00,
                "currency": "EUR",
                "counterparty": "DEUTSCHE BANK AG",
                "status": "SETTLED",
                "created_at": "2024-11-15T09:32:11Z",
                "updated_at": "2024-11-18T14:05:00Z",
            }
        ],
        "sql_executed": sql,
        "execution_time_ms": 42,
    }

    logger.info("query_executed", node=node_name, rows=len(mock_result["rows"]))
    return {
        "query_result": mock_result,
        "row_count": len(mock_result["rows"]),
        "steps": [_step(node_name, "query_executed", {
            "rows_returned": len(mock_result["rows"]),
            "execution_ms": mock_result["execution_time_ms"],
        })],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. DETAILS ANALYZER – interpret results, flag anomalies
# ─────────────────────────────────────────────────────────────────────────────

@with_retry(max_attempts=2, exceptions=(Exception,))
async def details_analyzer(state: TradeState) -> dict:
    node_name = "DetailsAnalyzer"
    logger.info("node_start", node=node_name)

    result = state.get("query_result", {})
    rows = result.get("rows", [])

    if not rows:
        return {
            "result_analysis": "No records found for the given criteria.",
            "confidence": 0.3,
            "steps": [_step(node_name, "analysis_complete", "no rows")],
        }

    system = SystemMessage(content=(
        "You are a trade data analyst. Analyze the following query result rows "
        "and produce a concise analysis including:\n"
        "  • Key fields and their significance\n"
        "  • Any anomalies or notable values\n"
        "  • Trade status summary\n"
        "Be factual, concise, and structured."
    ))
    human = HumanMessage(content=(
        f"User query: {state['query']}\n\n"
        f"Result rows ({len(rows)} row(s)):\n{json.dumps(rows, indent=2)}"
    ))

    response = await _llm().ainvoke([system, human])
    analysis = response.content.strip()

    confidence = 0.95 if len(rows) > 0 else 0.3

    return {
        "result_analysis": analysis,
        "confidence": confidence,
        "steps": [_step(node_name, "details_analyzed", analysis[:120] + "...")],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. RESULT SYNTHESIZER – produce final answer
# ─────────────────────────────────────────────────────────────────────────────

@with_retry(max_attempts=2, exceptions=(Exception,))
async def result_synthesizer(state: TradeState) -> dict:
    node_name = "ResultSynthesizer"
    logger.info("node_start", node=node_name)

    rows = state.get("query_result", {}).get("rows", [])
    analysis = state.get("result_analysis", "")

    system = SystemMessage(content=(
        "You are a financial data assistant. Synthesise the raw data and analysis "
        "into a clear, professional answer for the user. "
        "Include the most important fields in a readable format."
    ))
    human = HumanMessage(content=(
        f"Original question: {state['query']}\n\n"
        f"Data rows: {json.dumps(rows, indent=2)}\n\n"
        f"Analysis: {analysis}"
    ))

    response = await _llm().ainvoke([system, human])
    final_answer = response.content.strip()

    return {
        "answer": final_answer,
        "steps": [_step(node_name, "answer_synthesized", final_answer[:120] + "...")],
    }
