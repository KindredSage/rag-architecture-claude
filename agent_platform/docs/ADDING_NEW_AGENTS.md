# Adding a New Agent to the Platform

This guide walks through creating a new agent from scratch. The platform is designed so that new agents are automatically discovered by the Master Agent once registered.

## Step-by-step: Creating a "Risk Agent"

### 1. Create the directory structure

```
agents/risk/
    __init__.py         # Registration + descriptor
    state.py            # Agent state TypedDict
    graph.py            # LangGraph StateGraph
    nodes/
        __init__.py
        risk_scorer.py  # Your custom node
        exposure_calc.py
```

### 2. Define the state (`agents/risk/state.py`)

```python
from typing import Annotated, TypedDict

class RiskAgentState(TypedDict):
    # Input from Master Agent (always include these 3)
    user_query: str
    intent_analysis: dict
    context_overrides: dict

    # Your agent-specific state
    risk_scores: dict
    exposure_data: dict
    analysis: dict           # Required: master agent reads this
    artifacts: list[dict]    # Required: for file outputs

    # Control flow (always include these)
    execution_trace: Annotated[list[dict], list.__add__]
    needs_retry: bool
    retry_count: int
    error: str | None
```

### 3. Create your nodes (`agents/risk/nodes/risk_scorer.py`)

Follow this pattern for every node:

```python
import json, time, logging
from datetime import datetime, timezone
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a risk scoring specialist. ... Return ONLY valid JSON: ..."""

async def risk_scorer(state, *, llm) -> dict:
    start = time.perf_counter()
    try:
        response = await llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Query: {state['user_query']}"),
        ])
        # Parse, validate, return
        result = json.loads(response.content.strip())
        duration = (time.perf_counter() - start) * 1000
        return {
            "risk_scores": result,
            "execution_trace": [{
                "node": "risk_scorer",
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
            }],
        }
    except Exception as e:
        # Always handle errors gracefully
        return {
            "risk_scores": {},
            "error": str(e),
            "execution_trace": [{
                "node": "risk_scorer",
                "status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }],
        }
```

### 4. Build the graph (`agents/risk/graph.py`)

```python
from langgraph.graph import END, START, StateGraph
from agents.risk.state import RiskAgentState

def build_risk_agent_graph(*, settings, services):
    llm = services["llm"].get_model(fast=False)
    ch = services.get("clickhouse")

    async def _scorer(s):
        from agents.risk.nodes.risk_scorer import risk_scorer
        return await risk_scorer(s, llm=llm)

    # Add more nodes as needed...

    graph = StateGraph(RiskAgentState)
    graph.add_node("risk_scorer", _scorer)
    graph.add_edge(START, "risk_scorer")
    graph.add_edge("risk_scorer", END)
    return graph.compile()
```

### 5. Register the agent (`agents/risk/__init__.py`)

```python
from agents.registry import AgentRegistry
from models import AgentCapability, AgentDescriptor
from agents.risk.graph import build_risk_agent_graph

risk_descriptor = AgentDescriptor(
    agent_id="risk_agent",
    name="Risk Analysis Agent",
    description="Calculates risk scores, exposure, and VaR for trading positions.",
    capabilities=[AgentCapability.SQL_QUERY, AgentCapability.SUMMARY, AgentCapability.PLOTTING],
    domain_keywords=["risk", "VaR", "exposure", "stress test", "margin", "limit"],
    sub_agents=["risk_scorer"],
    example_queries=[
        "What is our current VaR?",
        "Show exposure by counterparty",
        "Run a stress test on the equity book",
    ],
    priority=8,
)

AgentRegistry.register(risk_descriptor, build_risk_agent_graph)
```

### 6. Add the import to `agents/__init__.py`

```python
import agents.risk  # noqa: F401   # <-- add this line
```

### 7. Done!

The Master Agent will automatically see your new agent in its routing context. No changes needed to the master agent, FastAPI endpoints, or any other code.

## Key Rules

1. Your graph factory MUST accept `(*, settings, services)` keyword arguments
2. Your output state MUST include `analysis` (dict) and `execution_trace` (list)
3. Every node MUST append to `execution_trace` for observability
4. Every node MUST handle exceptions and return a valid state (never crash)
5. LLM prompts MUST request JSON-only output (no markdown fences)
6. ClickHouse queries MUST go through the `ch_service` for safety validation

## Checklist

- [ ] State TypedDict defined with required fields
- [ ] All nodes handle LLM errors gracefully (try/except with fallback)
- [ ] JSON parsing handles markdown fences (```json...```)
- [ ] Execution trace appended at every node
- [ ] Graph factory accepts `settings` and `services`
- [ ] Descriptor has clear description and domain keywords
- [ ] Agent is imported in `agents/__init__.py`
- [ ] Tests written for each node
