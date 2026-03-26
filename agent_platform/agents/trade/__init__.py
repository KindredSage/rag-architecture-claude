"""
Trade Agent: Self-registers with the AgentRegistry on import.

To add a new agent, follow this pattern:
  1. Create agents/your_agent/ with graph.py, state.py, nodes/
  2. Define an AgentDescriptor
  3. Call AgentRegistry.register(descriptor, graph_factory)
  4. Import this module in agents/__init__.py
"""

from agents.registry import AgentRegistry
from models import AgentCapability, AgentDescriptor
from agents.trade.graph import build_trade_agent_graph

trade_descriptor = AgentDescriptor(
    agent_id="trade_agent",
    name="Trade Analysis Agent",
    description=(
        "Analyzes trading data in ClickHouse. Handles queries about trades, "
        "positions, PnL, volume, VWAP, fill rates, order flow, settlement, "
        "and counterparty analysis. Can generate SQL, execute queries, produce "
        "charts, export data, and send reports via email."
    ),
    capabilities=[
        AgentCapability.SQL_QUERY,
        AgentCapability.PLOTTING,
        AgentCapability.SUMMARY,
        AgentCapability.REPORT_GENERATION,
        AgentCapability.EMAIL,
        AgentCapability.DATA_EXPORT,
        AgentCapability.SCHEMA_EXPLORATION,
    ],
    domain_keywords=[
        "trade", "trades", "order", "fill", "execution", "PnL", "profit",
        "loss", "volume", "VWAP", "ticker", "symbol", "position", "book",
        "desk", "counterparty", "settlement", "notional", "slippage",
        "portfolio", "risk", "exposure", "hedge", "swap", "option", "future",
        "equity", "bond", "FX", "forex", "commodity", "derivative",
    ],
    sub_agents=[
        "trade_analyst", "query_analyst", "query_planner",
        "schema_analyzer", "query_builder", "query_validator",
        "query_executor", "details_analyzer",
    ],
    example_queries=[
        "What was the total PnL for desk A last month?",
        "Show me a chart of daily trading volume by ticker for Q3",
        "Which counterparties had the highest notional this week?",
        "Generate a PDF report of settlement failures in March",
        "Email the top 10 trades by value to risk@company.com",
        "What tables are available in the trading database?",
        "Compare fill rates across execution venues last quarter",
    ],
    priority=10,
    version="1.0.0",
)

AgentRegistry.register(trade_descriptor, build_trade_agent_graph)
