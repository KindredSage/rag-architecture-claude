"""
Analytics Agent: Trend analysis, anomaly detection, and data summarization.

Demonstrates a simpler agent with fewer nodes:
  trend_analyzer -> summary_generator -> END
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agents.registry import AgentRegistry
from models import AgentCapability, AgentDescriptor

logger = logging.getLogger(__name__)


class AnalyticsState(TypedDict):
    user_query: str
    intent_analysis: dict
    context_overrides: dict

    trend_data: dict
    analysis: dict
    artifacts: list[dict]

    execution_trace: Annotated[list[dict], list.__add__]
    needs_retry: bool
    retry_count: int
    error: str | None


async def trend_analyzer(state: AnalyticsState, *, ch_service, llm) -> dict:
    """Detect trends and anomalies in the data."""
    start = time.perf_counter()
    schema = ch_service.get_full_schema_context()

    # Generate analytical query
    try:
        sql_response = await llm.ainvoke([
            SystemMessage(content=(
                "You are a ClickHouse analytics expert. Generate a query to analyze trends. "
                "Include time-series grouping, running averages, or percent changes where applicable. "
                "Return ONLY the raw SQL, no markdown. Always include LIMIT 10000."
            )),
            HumanMessage(content=(
                f"Schema: {json.dumps(schema, default=str)[:4000]}\n"
                f"Analysis request: {state['user_query']}"
            )),
        ])
        sql = sql_response.content.strip()
        if sql.startswith("```"):
            sql = sql.split("\n", 1)[1] if "\n" in sql else sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        sql = sql.strip().rstrip(";")

        qr = ch_service.execute_query(sql)
        duration = (time.perf_counter() - start) * 1000

        return {
            "trend_data": {
                "sql": sql,
                "data": qr.get("data", []),
                "row_count": qr.get("row_count", 0),
                "success": qr.get("success", False),
                "columns": qr.get("columns", []),
            },
            "execution_trace": [{
                "node": "trend_analyzer",
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
                "output_summary": f"{qr.get('row_count', 0)} rows analyzed",
            }],
        }
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        return {
            "trend_data": {"success": False, "error": str(e)},
            "execution_trace": [{
                "node": "trend_analyzer", "status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1), "error": str(e),
            }],
        }


async def summary_generator(state: AnalyticsState, *, llm, plotting_tools=None) -> dict:
    """Generate narrative summary with optional charts."""
    start = time.perf_counter()
    trend_data = state.get("trend_data", {})
    data = trend_data.get("data", [])

    prompt = f"""Analyze this data and provide insights.

User Query: {state['user_query']}
Data ({len(data)} rows): {json.dumps(data[:20], default=str)[:4000]}
Columns: {json.dumps(trend_data.get('columns', []), default=str)}

Return ONLY valid JSON:
{{
  "narrative": "<2-3 paragraph analysis>",
  "key_findings": ["<insight1>", "<insight2>"],
  "anomalies": ["<any unusual patterns>"],
  "data_summary": {{"metric": "value"}},
  "visualization_recommendations": [
    {{"chart_type": "<type>", "x_axis": "<col>", "y_axis": "<col>", "title": "<title>"}}
  ],
  "follow_up_questions": ["<q1>", "<q2>"],
  "confidence": 0.8
}}"""

    try:
        response = await llm.ainvoke([
            SystemMessage(content="You are a data analyst. Return ONLY valid JSON."),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        analysis = json.loads(raw.strip())
    except Exception as e:
        analysis = {
            "narrative": f"Analysis of {len(data)} data points. Error: {e}",
            "key_findings": [], "confidence": 0.3,
        }

    # Generate charts
    artifacts: list[dict] = []
    if plotting_tools and data:
        for rec in analysis.get("visualization_recommendations", [])[:1]:
            try:
                chart_tool = next((t for t in plotting_tools if "plotly" in t.name.lower()), None)
                if chart_tool:
                    cols = list(data[0].keys()) if data else []
                    result = chart_tool.invoke({
                        "chart_type": rec.get("chart_type", "line"),
                        "data": data[:2000],
                        "x": rec.get("x_axis", cols[0] if cols else "x"),
                        "y": rec.get("y_axis", cols[1] if len(cols) > 1 else cols[0] if cols else "y"),
                        "title": rec.get("title", "Analysis"),
                    })
                    if result.get("html_path"):
                        artifacts.append({
                            "type": "chart_html", "name": "analytics_chart.html",
                            "path": result["html_path"],
                        })
            except Exception as e:
                logger.warning("Analytics chart failed: %s", e)

    analysis["query_results"] = {"data": data, "row_count": len(data), "success": True}
    duration = (time.perf_counter() - start) * 1000

    return {
        "analysis": analysis,
        "artifacts": artifacts,
        "execution_trace": [{
            "node": "summary_generator",
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round(duration, 1),
            "output_summary": f"{len(analysis.get('key_findings', []))} findings",
        }],
    }


def build_analytics_agent_graph(*, settings, services):
    llm_service = services["llm"]
    llm = llm_service.get_model(fast=False)
    ch = services["clickhouse"]

    from tools.plotting_tools import create_plotting_tools
    plot_tools = create_plotting_tools(settings.artifact_dir)

    async def _trend(s):
        return await trend_analyzer(s, ch_service=ch, llm=llm)

    async def _summary(s):
        return await summary_generator(s, llm=llm, plotting_tools=plot_tools)

    graph = StateGraph(AnalyticsState)
    graph.add_node("trend_analyzer", _trend)
    graph.add_node("summary_generator", _summary)
    graph.add_edge(START, "trend_analyzer")
    graph.add_edge("trend_analyzer", "summary_generator")
    graph.add_edge("summary_generator", END)
    return graph.compile()


analytics_descriptor = AgentDescriptor(
    agent_id="analytics_agent",
    name="Analytics & Insights Agent",
    description=(
        "Performs trend analysis, anomaly detection, and data summarization. "
        "Best for exploratory analysis, pattern identification, and generating "
        "high-level insights from trading or operational data."
    ),
    capabilities=[
        AgentCapability.SQL_QUERY,
        AgentCapability.SUMMARY,
        AgentCapability.ANOMALY_DETECTION,
        AgentCapability.PLOTTING,
    ],
    domain_keywords=[
        "trend", "anomaly", "pattern", "insight", "analyze", "analysis",
        "compare", "correlation", "growth", "decline", "unusual",
        "distribution", "statistics", "average", "median", "percentile",
    ],
    sub_agents=["trend_analyzer", "summary_generator"],
    example_queries=[
        "Are there any anomalies in last week's trading volume?",
        "Show me the trend of daily PnL over the past 6 months",
        "What are the key statistics for EUR/USD trades this quarter?",
    ],
    priority=7,
    version="1.0.0",
)

AgentRegistry.register(analytics_descriptor, build_analytics_agent_graph)
