"""
Chart generation tools using Plotly and Matplotlib.

Charts are saved as HTML (interactive) and PNG (static) artifacts.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def create_plotting_tools(artifact_dir: str) -> list:
    """Build plotting tools with artifact directory injected."""

    os.makedirs(artifact_dir, exist_ok=True)

    @tool
    def generate_plotly_chart(
        chart_type: str,
        data: list[dict[str, Any]],
        x: str,
        y: str | list[str],
        title: str = "Chart",
        color: str | None = None,
        layout_overrides: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Generate an interactive Plotly chart.

        Args:
            chart_type: One of bar, line, scatter, pie, heatmap, histogram,
                        box, candlestick, treemap, funnel, area.
            data: List of row dicts (query results).
            x: Column name for x-axis.
            y: Column name(s) for y-axis.
            title: Chart title.
            color: Optional column for color grouping.
            layout_overrides: Optional Plotly layout dict overrides.

        Returns:
            Dict with html_path, png_path, and plotly_json.
        """
        import plotly.express as px
        import plotly.graph_objects as go
        import plotly.io as pio
        import pandas as pd

        df = pd.DataFrame(data)
        if df.empty:
            return {"error": "No data to plot", "html_path": "", "png_path": ""}

        chart_id = str(uuid.uuid4())[:8]
        chart_map = {
            "bar": px.bar,
            "line": px.line,
            "scatter": px.scatter,
            "pie": px.pie,
            "histogram": px.histogram,
            "box": px.box,
            "area": px.area,
            "funnel": px.funnel,
            "treemap": px.treemap,
        }

        kwargs: dict[str, Any] = {"data_frame": df, "title": title}

        if chart_type == "pie":
            kwargs["names"] = x
            kwargs["values"] = y if isinstance(y, str) else y[0]
        elif chart_type == "heatmap":
            fig = go.Figure(
                data=go.Heatmap(
                    z=df[y if isinstance(y, str) else y[0]].values.reshape(-1, len(df[x].unique())),
                    x=df[x].unique().tolist(),
                    y=df[x].unique().tolist(),
                )
            )
            fig.update_layout(title=title)
        elif chart_type == "candlestick":
            # Expects columns: open, high, low, close
            fig = go.Figure(
                data=go.Candlestick(
                    x=df[x],
                    open=df.get("open", df.iloc[:, 1]),
                    high=df.get("high", df.iloc[:, 2]),
                    low=df.get("low", df.iloc[:, 3]),
                    close=df.get("close", df.iloc[:, 4]),
                )
            )
            fig.update_layout(title=title)
        else:
            func = chart_map.get(chart_type, px.bar)
            kwargs["x"] = x
            kwargs["y"] = y
            if color:
                kwargs["color"] = color
            fig = None

        if "fig" not in dir() or fig is None:
            func = chart_map.get(chart_type, px.bar)
            fig = func(**kwargs)

        if layout_overrides:
            fig.update_layout(**layout_overrides)

        # Save artifacts
        html_path = os.path.join(artifact_dir, f"chart_{chart_id}.html")
        png_path = os.path.join(artifact_dir, f"chart_{chart_id}.png")

        fig.write_html(html_path, include_plotlyjs="cdn")

        try:
            fig.write_image(png_path, width=1200, height=600, scale=2)
        except Exception as e:
            logger.warning("PNG export failed (kaleido may not be installed): %s", e)
            png_path = ""

        plotly_json = json.loads(pio.to_json(fig))

        logger.info("Chart generated: %s (%s)", chart_type, chart_id)
        return {
            "html_path": html_path,
            "png_path": png_path,
            "plotly_json": plotly_json,
            "chart_id": chart_id,
        }

    @tool
    def generate_matplotlib_chart(
        chart_type: str,
        data: list[dict[str, Any]],
        x: str,
        y: str,
        title: str = "Chart",
    ) -> dict[str, str]:
        """Generate a static chart using matplotlib. Returns PNG path.
        Fallback when Plotly is not suitable."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.DataFrame(data)
        if df.empty:
            return {"error": "No data", "png_path": ""}

        chart_id = str(uuid.uuid4())[:8]
        fig, ax = plt.subplots(figsize=(12, 6))

        if chart_type == "bar":
            ax.bar(df[x], df[y])
        elif chart_type == "line":
            ax.plot(df[x], df[y])
        elif chart_type == "scatter":
            ax.scatter(df[x], df[y])
        else:
            ax.bar(df[x], df[y])

        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        png_path = os.path.join(artifact_dir, f"mpl_{chart_id}.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return {"png_path": png_path, "chart_id": chart_id}

    return [generate_plotly_chart, generate_matplotlib_chart]
