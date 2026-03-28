"""
Reporting Agent: Generates formatted reports from data queries.

This agent demonstrates extensibility. It reuses some patterns from the
trade agent but focuses on report generation workflows:
  report_planner -> data_fetcher -> chart_generator -> document_builder -> email_dispatcher
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from services.llm_invoke import invoke_llm
from langgraph.graph import END, START, StateGraph

from agents.registry import AgentRegistry
from models import AgentCapability, AgentDescriptor

logger = logging.getLogger(__name__)


# ── State ────────────────────────────────────────────────────────


class ReportingState(TypedDict):
    user_query: str
    intent_analysis: dict
    context_overrides: dict

    report_plan: dict
    fetched_data: dict
    charts: list[dict]
    document_path: str
    email_result: dict
    analysis: dict
    artifacts: list[dict]

    execution_trace: Annotated[list[dict], list.__add__]
    error: str | None
    needs_retry: bool
    retry_count: int


# ── Nodes ────────────────────────────────────────────────────────


async def report_planner(state: ReportingState, *, llm) -> dict:
    """Plan the report structure: sections, data needed, charts, format."""
    start = time.perf_counter()

    prompt = f"""Plan a data report based on this request.

User Query: {state['user_query']}
Intent: {json.dumps(state.get('intent_analysis', {}), default=str)}

Return ONLY valid JSON:
{{
  "title": "<report title>",
  "format": "<pdf|docx|xlsx>",
  "sections": [
    {{
      "heading": "<section heading>",
      "data_query_description": "<what data is needed>",
      "chart_type": "<bar|line|pie|none>",
      "narrative_needed": true
    }}
  ],
  "recipients": ["<email addresses if email was requested>"],
  "time_period": "<time range for data>"
}}"""

    try:
        response = await invoke_llm(llm, [
            SystemMessage(content="You are a report planner. Return ONLY valid JSON."),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        plan = json.loads(raw.strip())
        duration = (time.perf_counter() - start) * 1000

        return {
            "report_plan": plan,
            "execution_trace": [{
                "node": "report_planner",
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
                "output_summary": f"Report: {plan.get('title')}, {len(plan.get('sections', []))} sections",
            }],
        }
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        return {
            "report_plan": {"title": "Report", "format": "pdf", "sections": []},
            "error": str(e),
            "execution_trace": [{
                "node": "report_planner", "status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1), "error": str(e),
            }],
        }


async def data_fetcher(state: ReportingState, *, ch_service, llm) -> dict:
    """Generate and execute queries for each report section."""
    start = time.perf_counter()
    plan = state.get("report_plan", {})
    sections = plan.get("sections", [])
    schema = ch_service.get_full_schema_context()

    results = {}
    for i, section in enumerate(sections):
        query_desc = section.get("data_query_description", "")
        if not query_desc:
            continue

        try:
            sql_response = await invoke_llm(llm, [
                SystemMessage(content=(
                    "You are a ClickHouse SQL expert. Generate a SELECT query. "
                    "Return ONLY the raw SQL, no explanation, no markdown fences. "
                    "Always include LIMIT 10000."
                )),
                HumanMessage(content=(
                    f"Schema: {json.dumps(schema, default=str)[:4000]}\n"
                    f"Data needed: {query_desc}\n"
                    f"Time period: {plan.get('time_period', 'last 30 days')}"
                )),
            ])
            sql = sql_response.content.strip().rstrip(";").strip()
            if sql.startswith("```"):
                sql = sql.split("\n", 1)[1] if "\n" in sql else sql[3:]
            if sql.endswith("```"):
                sql = sql[:-3]
            sql = sql.strip()

            qr = ch_service.execute_query(sql)
            results[f"section_{i}"] = {
                "heading": section.get("heading", f"Section {i+1}"),
                "sql": sql,
                "data": qr.get("data", []),
                "row_count": qr.get("row_count", 0),
                "success": qr.get("success", False),
            }
        except Exception as e:
            logger.error("Data fetch failed for section %d: %s", i, e)
            results[f"section_{i}"] = {"heading": section.get("heading"), "error": str(e)}

    duration = (time.perf_counter() - start) * 1000
    return {
        "fetched_data": results,
        "execution_trace": [{
            "node": "data_fetcher",
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round(duration, 1),
            "output_summary": f"Fetched data for {len(results)} sections",
        }],
    }


async def document_builder(
    state: ReportingState, *, llm, export_tools, plotting_tools
) -> dict:
    """Build the report document with charts and narrative."""
    start = time.perf_counter()
    plan = state.get("report_plan", {})
    data = state.get("fetched_data", {})
    artifacts: list[dict] = []
    sections_content = []

    for key, section_data in data.items():
        heading = section_data.get("heading", key)
        rows = section_data.get("data", [])

        # Generate narrative
        try:
            narr_response = await invoke_llm(llm, [
                SystemMessage(content="Summarize this data section in 2-3 sentences. Be concise and analytical."),
                HumanMessage(content=f"Section: {heading}\nData ({len(rows)} rows): {json.dumps(rows[:10], default=str)}"),
            ])
            narrative = narr_response.content.strip()
        except Exception:
            narrative = f"This section contains {len(rows)} data points."

        sections_content.append({"heading": heading, "body": narrative})

        # Generate chart if applicable
        plan_sections = plan.get("sections", [])
        idx = int(key.split("_")[1]) if "_" in key else 0
        if idx < len(plan_sections) and plan_sections[idx].get("chart_type", "none") != "none" and rows:
            try:
                chart_tool = next((t for t in plotting_tools if "plotly" in t.name.lower()), None)
                if chart_tool and rows:
                    cols = list(rows[0].keys())
                    result = chart_tool.invoke({
                        "chart_type": plan_sections[idx]["chart_type"],
                        "data": rows[:1000],
                        "x": cols[0],
                        "y": cols[1] if len(cols) > 1 else cols[0],
                        "title": heading,
                    })
                    if result.get("png_path"):
                        artifacts.append({
                            "type": "chart_png", "name": f"{heading}.png",
                            "path": result["png_path"], "mime_type": "image/png",
                        })
            except Exception as e:
                logger.warning("Chart generation failed: %s", e)

    # Build document
    fmt = plan.get("format", "pdf")
    chart_paths = [a["path"] for a in artifacts if a["type"] == "chart_png"]

    try:
        if fmt == "docx":
            doc_tool = next((t for t in export_tools if "docx" in t.name.lower()), None)
        else:
            doc_tool = next((t for t in export_tools if "pdf" in t.name.lower()), None)

        if doc_tool:
            result = doc_tool.invoke({
                "title": plan.get("title", "Report"),
                "sections": sections_content,
                "include_charts": chart_paths,
            })
            if result.get("path"):
                artifacts.append({
                    "type": fmt, "name": f"{plan.get('title', 'report')}.{fmt}",
                    "path": result["path"],
                    "size_bytes": result.get("size_bytes", 0),
                })
    except Exception as e:
        logger.error("Document build failed: %s", e)

    duration = (time.perf_counter() - start) * 1000
    return {
        "document_path": artifacts[-1]["path"] if artifacts else "",
        "artifacts": artifacts,
        "analysis": {
            "narrative": "\n\n".join(s.get("body", "") for s in sections_content),
            "key_findings": [s.get("heading", "") for s in sections_content],
            "confidence": 0.8,
        },
        "execution_trace": [{
            "node": "document_builder",
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round(duration, 1),
            "output_summary": f"Built {fmt} with {len(sections_content)} sections, {len(artifacts)} artifacts",
        }],
    }


async def email_dispatcher(state: ReportingState, *, email_tools) -> dict:
    """Send report via email if recipients were specified."""
    start = time.perf_counter()
    plan = state.get("report_plan", {})
    recipients = plan.get("recipients", [])

    if not recipients or not email_tools:
        duration = (time.perf_counter() - start) * 1000
        return {
            "email_result": {"skipped": True, "reason": "No recipients or email not configured"},
            "execution_trace": [{
                "node": "email_dispatcher", "status": "skipped",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
            }],
        }

    attachments = [a["path"] for a in state.get("artifacts", []) if a.get("path")]
    try:
        send_tool = email_tools[0]
        result = send_tool.invoke({
            "to": recipients,
            "subject": plan.get("title", "Report"),
            "body": state.get("analysis", {}).get("narrative", "Please find the attached report."),
            "body_format": "html",
            "attachments": attachments,
        })
        duration = (time.perf_counter() - start) * 1000
        return {
            "email_result": result,
            "execution_trace": [{
                "node": "email_dispatcher",
                "status": "completed" if result.get("success") else "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
                "output_summary": f"Email {'sent' if result.get('success') else 'failed'} to {recipients}",
            }],
        }
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        return {
            "email_result": {"success": False, "error": str(e)},
            "execution_trace": [{
                "node": "email_dispatcher", "status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1), "error": str(e),
            }],
        }


# ── Graph Assembly ───────────────────────────────────────────────


def build_reporting_agent_graph(*, settings, services):
    llm_service = services["llm"]
    llm = llm_service.get_model(fast=False)
    llm_fast = llm_service.get_model(fast=True)
    ch = services["clickhouse"]

    from tools.plotting_tools import create_plotting_tools
    from tools.export_tools import create_export_tools
    from tools.email_tools import create_email_tools

    plot_tools = create_plotting_tools(settings.artifact_dir)
    exp_tools = create_export_tools(settings.artifact_dir)
    mail_tools = create_email_tools(settings)

    async def _plan(s):
        return await report_planner(s, llm=llm_fast)

    async def _fetch(s):
        return await data_fetcher(s, ch_service=ch, llm=llm)

    async def _build(s):
        return await document_builder(
            s, llm=llm, export_tools=exp_tools, plotting_tools=plot_tools
        )

    async def _email(s):
        return await email_dispatcher(s, email_tools=mail_tools)

    graph = StateGraph(ReportingState)
    graph.add_node("report_planner", _plan)
    graph.add_node("data_fetcher", _fetch)
    graph.add_node("document_builder", _build)
    graph.add_node("email_dispatcher", _email)

    graph.add_edge(START, "report_planner")
    graph.add_edge("report_planner", "data_fetcher")
    graph.add_edge("data_fetcher", "document_builder")
    graph.add_edge("document_builder", "email_dispatcher")
    graph.add_edge("email_dispatcher", END)

    return graph.compile()


# ── Registration ─────────────────────────────────────────────────

reporting_descriptor = AgentDescriptor(
    agent_id="reporting_agent",
    name="Report Generation Agent",
    description=(
        "Generates formatted reports (PDF, DOCX) from ClickHouse data. "
        "Plans report structure, fetches data, generates charts, builds "
        "documents, and optionally emails them to recipients."
    ),
    capabilities=[
        AgentCapability.REPORT_GENERATION,
        AgentCapability.PLOTTING,
        AgentCapability.EMAIL,
        AgentCapability.DATA_EXPORT,
        AgentCapability.SUMMARY,
    ],
    domain_keywords=[
        "report", "PDF", "document", "generate report", "weekly report",
        "monthly report", "summary report", "send report", "email report",
        "dashboard", "executive summary", "brief",
    ],
    sub_agents=[
        "report_planner", "data_fetcher", "chart_generator",
        "document_builder", "email_dispatcher",
    ],
    example_queries=[
        "Generate a monthly PnL report as PDF",
        "Create a weekly trading summary and email it to team@company.com",
        "Build a risk exposure report with charts",
    ],
    priority=8,
    version="1.0.0",
)

AgentRegistry.register(reporting_descriptor, build_reporting_agent_graph)
