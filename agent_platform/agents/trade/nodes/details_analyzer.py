"""
Details Analyzer: Interprets query results and produces actionable analysis.

This is the final analytical node. It:
  1. Produces a natural-language narrative from the raw data
  2. Identifies key findings and insights
  3. Recommends visualizations based on data shape
  4. Generates charts if the user requested plotting
  5. Suggests follow-up questions
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from services.llm_invoke import invoke_llm

from agents.trade.state import TradeAgentState

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """You are a senior trade data analyst. Given the query results and the user's
original question, provide a comprehensive analysis.

Produce ONLY a valid JSON object:
{{
  "narrative": "<Natural language answer to the user's question, 2-4 paragraphs>",
  "key_findings": ["<top 3-5 bullet-point insights>"],
  "data_summary": {{
    "total_records": <int>,
    "key_metrics": {{"<metric_name>": "<value with units>"}},
    "trends": "<notable patterns if time series data>"
  }},
  "visualization_recommendations": [
    {{
      "chart_type": "<bar|line|scatter|heatmap|pie|treemap|candlestick|area|histogram|box>",
      "x_axis": "<column name>",
      "y_axis": "<column name or list>",
      "color": "<optional grouping column or null>",
      "title": "<descriptive chart title>",
      "reason": "<why this chart type is appropriate>"
    }}
  ],
  "follow_up_questions": ["<2-3 natural follow-up questions>"],
  "caveats": ["<data quality notes, limitations, assumptions>"],
  "confidence": <0.0-1.0>
}}

Consider the data shape:
- Time series data -> recommend line charts
- Categorical comparisons -> bar charts
- Distributions -> histograms or box plots
- Part-of-whole -> pie/treemap
- Two numeric dimensions -> scatter plots
- Financial OHLC data -> candlestick charts"""


async def details_analyzer(
    state: TradeAgentState,
    *,
    llm,
    plotting_tools=None,
    export_tools=None,
    email_tools=None,
) -> dict:
    start = time.perf_counter()
    query_results = state.get("query_results", {})
    intent = state.get("intent_analysis", {})
    desired_output = intent.get("desired_output", "table")

    if not query_results.get("success", False):
        duration = (time.perf_counter() - start) * 1000
        error = query_results.get("error", "No results available")
        return {
            "analysis": {
                "narrative": f"The query could not be completed: {error}",
                "key_findings": [],
                "confidence": 0.0,
            },
            "execution_trace": [{
                "node": "details_analyzer",
                "status": "skipped",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
                "output_summary": f"Skipped: {error}",
            }],
        }

    data = query_results.get("data", [])
    columns = query_results.get("columns", [])

    # Prepare data summary for LLM (truncate for context window)
    sample_data = data[:20] if len(data) > 20 else data
    prompt = f"""User Query: {state['user_query']}

Query Results:
- Total Rows: {query_results.get('row_count', 0)}
- Columns: {json.dumps(columns, default=str)}
- Execution Time: {query_results.get('execution_time_ms', 0):.0f}ms
- Truncated: {query_results.get('truncated', False)}

Sample Data (first {len(sample_data)} rows):
{json.dumps(sample_data, default=str, indent=2)[:6000]}

SQL Used: {state.get('generated_sql', 'N/A')}
Desired Output: {desired_output}"""

    try:
        response = await invoke_llm(llm, [
            SystemMessage(content=ANALYSIS_PROMPT),
            HumanMessage(content=prompt),
        ])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]

        analysis = json.loads(raw.strip())

    except Exception as e:
        logger.error("Analysis LLM call failed: %s", e)
        analysis = {
            "narrative": f"Query returned {len(data)} rows. Analysis generation encountered an error: {e}",
            "key_findings": [f"Query returned {len(data)} rows"],
            "data_summary": {"total_records": len(data)},
            "visualization_recommendations": [],
            "follow_up_questions": [],
            "caveats": ["Automated analysis was unavailable"],
            "confidence": 0.4,
        }

    # --- Post-processing: Generate artifacts based on desired output ---
    artifacts: list[dict] = []
    trace_extras: list[dict] = []

    # Chart generation
    viz_recs = analysis.get("visualization_recommendations", [])
    if desired_output in ("chart", "report") and viz_recs and plotting_tools and data:
        for rec in viz_recs[:2]:  # Max 2 charts
            try:
                gen_tool = next(
                    (t for t in plotting_tools if "plotly" in t.name.lower()),
                    plotting_tools[0] if plotting_tools else None,
                )
                if gen_tool:
                    chart_result = gen_tool.invoke({
                        "chart_type": rec.get("chart_type", "bar"),
                        "data": data[:5000],  # Cap for performance
                        "x": rec.get("x_axis", columns[0]["name"] if columns else "x"),
                        "y": rec.get("y_axis", columns[1]["name"] if len(columns) > 1 else "y"),
                        "title": rec.get("title", "Chart"),
                        "color": rec.get("color"),
                    })

                    if chart_result.get("html_path"):
                        artifacts.append({
                            "type": "chart_html",
                            "name": f"{rec.get('title', 'chart')}.html",
                            "path": chart_result["html_path"],
                            "mime_type": "text/html",
                        })
                    if chart_result.get("png_path"):
                        artifacts.append({
                            "type": "chart_png",
                            "name": f"{rec.get('title', 'chart')}.png",
                            "path": chart_result["png_path"],
                            "mime_type": "image/png",
                        })
                    trace_extras.append({
                        "node": "details_analyzer:chart",
                        "status": "completed",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "output_summary": f"Chart generated: {rec.get('chart_type')}",
                    })
            except Exception as e:
                logger.error("Chart generation failed: %s", e)
                trace_extras.append({
                    "node": "details_analyzer:chart",
                    "status": "error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": str(e),
                })

    # Data export
    if desired_output in ("export", "report") and export_tools and data:
        try:
            xlsx_tool = next(
                (t for t in export_tools if "xlsx" in t.name.lower()),
                None,
            )
            if xlsx_tool:
                export_result = xlsx_tool.invoke({
                    "data": data,
                    "title": "trade_export",
                    "sheet_name": "Results",
                })
                if export_result.get("path"):
                    artifacts.append({
                        "type": "xlsx",
                        "name": "trade_export.xlsx",
                        "path": export_result["path"],
                        "size_bytes": export_result.get("size_bytes", 0),
                        "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    })
        except Exception as e:
            logger.error("Export failed: %s", e)

    # Email
    if desired_output == "email" and email_tools:
        # Email sending is deferred to merge_results or a dedicated step.
        # We just flag it here.
        analysis["email_requested"] = True
        analysis["email_attachments"] = [a["path"] for a in artifacts if a.get("path")]

    duration = (time.perf_counter() - start) * 1000

    logger.info(
        "Analysis complete: %d findings, %d viz recs, %d artifacts (%.0fms)",
        len(analysis.get("key_findings", [])),
        len(viz_recs),
        len(artifacts),
        duration,
    )

    return {
        "analysis": analysis,
        "artifacts": artifacts,
        "execution_trace": [{
            "node": "details_analyzer",
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round(duration, 1),
            "output_summary": f"{len(analysis.get('key_findings', []))} findings, {len(artifacts)} artifacts",
        }] + trace_extras,
    }
