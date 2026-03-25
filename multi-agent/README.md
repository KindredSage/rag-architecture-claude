# Multi-Agent Orchestration System

Production-grade system where **LangGraph agents** are exposed via **FastAPI APIs**.  
A Master Agent uses LLM reasoning (not rule-based logic) to dynamically route queries to specialised sub-agents.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     FastAPI Layer                        │
│   GET /agents        POST /execute        GET /health    │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│               Master Agent  (LangGraph)                  │
│                                                          │
│   [Input] → [IntentAnalyzer] → [AgentSelector(LLM)]     │
│                    ↓                    ↓                │
│            [DirectAnswer]     [AgentExecutor]            │
│                    ↓                    ↓                │
│                    └──── [ResponseSynthesizer] ──► OUT   │
└──────────────────────────────────────────────────────────┘
                            │
                   (registry lookup)
                            │
┌───────────────────────────▼─────────────────────────────┐
│               Trade Agent  (LangGraph)                   │
│                                                          │
│  QueryAnalyst → QueryPlanner → SchemaAnalyzer            │
│       → QueryBuilder → QueryValidator                    │
│             ↓ (conditional)                              │
│       QueryExecutor → DetailsAnalyzer → ResultSynthesizer│
└──────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
multi_agent_system/
├── app/
│   ├── main.py                        # FastAPI app + lifespan startup
│   ├── core/
│   │   ├── config.py                  # Pydantic-Settings configuration
│   │   └── logging.py                 # Structlog setup
│   ├── api/
│   │   ├── models/
│   │   │   └── schemas.py             # Request / Response Pydantic models
│   │   └── routes/
│   │       └── routes.py              # GET /agents  +  POST /execute
│   ├── registry/
│   │   └── agent_registry.py          # Plug-and-play singleton registry
│   ├── agents/
│   │   ├── base.py                    # BaseAgent abstract class
│   │   ├── master/
│   │   │   ├── state.py               # MasterState TypedDict
│   │   │   ├── nodes.py               # IntentAnalyzer, AgentSelector, etc.
│   │   │   └── graph.py               # Master LangGraph (compiled)
│   │   └── trade/
│   │       ├── state.py               # TradeState TypedDict
│   │       ├── nodes.py               # 8 sub-agent nodes
│   │       ├── graph.py               # Trade LangGraph (compiled)
│   │       └── agent.py               # TradeAgent(BaseAgent) wrapper
│   └── utils/
│       └── retry.py                   # Async retry decorator
├── tests/
│   └── test_pipeline.py               # Full integration tests (mocked LLM)
├── examples/
│   ├── api_calls.py                   # Live API demo script
│   └── sample_response.json           # Expected response for trade query
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
cd multi_agent_system
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Test the API

```bash
# List agents
curl http://localhost:8000/agents | python -m json.tool

# Execute trade query
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"query": "Give me trade details for file id 999 from My_Table", "context": {}}' \
  | python -m json.tool
```

---

## API Reference

### `GET /agents`
Returns all registered agents with their capabilities.

**Response:**
```json
[
  {
    "name": "trade_agent",
    "description": "Handles all trade data queries...",
    "capabilities": [
      "retrieve trade details by file_id",
      "filter trades by date range",
      ...
    ]
  }
]
```

---

### `POST /execute`
Routes a query through the Master Agent and returns a structured response.

**Request:**
```json
{
  "query": "Give me trade details for file id 999 from My_Table",
  "context": {
    "user_id": "analyst_01"
  }
}
```

**Response:**
```json
{
  "answer": "Here are the trade details for file ID 999...",
  "agent_used": "trade_agent",
  "steps": [
    {"node": "IntentAnalyzer", "action": "extracted_intent", "output": "retrieve trade details"},
    {"node": "AgentSelector",  "action": "agent_selected",   "output": {"agent": "trade_agent", "reason": "..."}},
    {"node": "QueryAnalyst",   "action": "query_analyzed",   "output": {"table_name": "My_Table", "file_id": "999"}},
    ...
  ],
  "confidence": 0.95
}
```

---

## Running Tests (no API key needed)

All LLM calls are mocked:

```bash
python -m pytest tests/test_pipeline.py -v
# or
python tests/test_pipeline.py
```

---

## Adding a New Agent

1. Create `app/agents/my_agent/` with `state.py`, `nodes.py`, `graph.py`, `agent.py`
2. Implement `class MyAgent(BaseAgent)` with `name`, `description`, `capabilities`, `execute()`
3. In `app/main.py`, add one line inside `lifespan()`:
   ```python
   registry.register(MyAgent())
   ```

That's it. The Master Agent will **automatically discover** the new agent via the registry and include it in LLM routing decisions.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| LLM-based routing | Master Agent feeds agent manifest to LLM for selection — no hardcoded if/else |
| Registry singleton | Plug-and-play: register agents at startup, Master Agent auto-discovers them |
| Separate state per graph | `MasterState` and `TradeState` are decoupled; agents are independently testable |
| Conditional edges in trade graph | Validation failures short-circuit to `ResultSynthesizer`, skipping DB execution |
| Retry decorator | All LLM-calling nodes wrapped with exponential-backoff retry via `@with_retry` |
| Lifespan over `on_event` | Modern FastAPI pattern; ensures async resources are properly managed |
| Structlog | JSON-structured logs carry `trace_id` context for end-to-end request tracing |
