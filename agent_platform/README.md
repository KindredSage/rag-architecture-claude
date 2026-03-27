# Agent Platform

Production-grade multi-agent orchestration system built with LangGraph, FastAPI, ClickHouse, and PostgreSQL.

## Table of Contents

1. [Architecture](#architecture)
2. [Quick Start](#quick-start)
3. [API Reference](#api-reference)
4. [Agent Pipeline Flow](#agent-pipeline-flow)
5. [Session Management & Multi-Server Scalability](#session-management--multi-server-scalability)
6. [Adding New Agents](#adding-new-agents)
7. [Tools & Capabilities](#tools--capabilities)
8. [Security Model](#security-model)
9. [Testing](#testing)
10. [Deployment](#deployment)
11. [Configuration Reference](#configuration-reference)

---

## Architecture

```
                         Load Balancer (nginx)
                          |       |       |
                   +------+  +----+  +----+------+
                   |         |       |           |
               Server-1  Server-2  Server-3     ...
               (FastAPI)  (FastAPI) (FastAPI)
                   |         |       |
                   +---------+-------+-----------+
                   |                             |
              PostgreSQL                      Redis
         (sessions, runs,              (cache, rate limits,
          checkpoints,                  schema cache,
          message history)              query dedup)
                   |
              ClickHouse
         (trading data, analytics)
```

Each server is stateless. All state lives in PostgreSQL (sessions, runs, history) and Redis (cache). Any server can serve any session, any request, at any time.

### Component Overview

```
FastAPI App
  |
  +-- /health              GET   -> health check
  +-- /agents              POST  -> list registered agents
  +-- /execute             POST  -> run query (sync)
  +-- /execute/stream      POST  -> run query (SSE streaming)
  +-- /execute/{run_id}    GET   -> poll run result
  +-- /sessions/{id}       GET   -> session history
  +-- /sessions/{id}       DELETE-> close session
  |
  +-- Master Agent (LangGraph StateGraph)
  |     |
  |     +-- classify_intent   (LLM: parse query intent)
  |     +-- select_agents     (LLM: route to best agent)
  |     +-- execute_agents    (invoke sub-graph(s))
  |     +-- merge_results     (LLM: synthesize final answer)
  |
  +-- Trade Agent (Sub-Graph, 8 nodes)
  |     +-- trade_analyst     -> query_analyst -> query_planner
  |     +-- schema_analyzer   -> query_builder -> query_validator
  |     +-- query_executor    -> details_analyzer
  |     +-- (retry loop: validator -> builder on failure, max 3)
  |
  +-- Reporting Agent (Sub-Graph, 5 nodes)
  |     +-- report_planner -> data_fetcher -> chart_generator
  |     +-- document_builder -> email_dispatcher
  |
  +-- Analytics Agent (Sub-Graph, 4 nodes)
        +-- trend_analyzer -> anomaly_detector
        +-- summary_generator -> insight_ranker
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Redis 7+
- ClickHouse 23+ (for data queries)
- An LLM API key (OpenAI, Azure, or Anthropic)

### Local Development

```bash
# 1. Clone and setup
cd agent_platform
cp .env.example .env
# Edit .env with your LLM API key and database credentials

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start infrastructure (if using Docker)
docker compose up postgres redis -d

# 4. Run the app
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 5. Test
curl http://localhost:8000/health
```

### Docker (Full Stack)

```bash
# Single server + all infrastructure
docker compose --profile full up

# Multi-server with load balancer
docker compose --profile multi-server up
```

---

## API Reference

### POST /agents

List all registered agents and their capabilities.

```bash
curl -X POST http://localhost:8000/agents
```

Response:
```json
[
  {
    "agent_id": "trade_agent",
    "name": "Trade Analysis Agent",
    "description": "Analyzes trading data in ClickHouse...",
    "capabilities": ["sql_query", "plotting", "summary", "email"],
    "sub_agents": ["trade_analyst", "query_analyst", "query_planner", "schema_analyzer", "query_builder", "query_validator", "query_executor", "details_analyzer"],
    "example_queries": ["What was the total PnL for desk A last month?"],
    "version": "1.0.0",
    "enabled": true
  }
]
```

### POST /execute

Execute a user query through the master agent.

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me PnL by desk for the last 30 days",
    "session_id": null,
    "user_id": "user-123",
    "preferences": {"chart_theme": "dark"},
    "agent_id": null,
    "stream": false
  }'
```

Response:
```json
{
  "run_id": "uuid",
  "session_id": "uuid",
  "status": "completed",
  "answer": "Desk Alpha leads with $2.3M in PnL...",
  "data": [{"desk": "Alpha", "total_pnl": 2300000}, ...],
  "visualizations": [{"chart_type": "bar", "title": "PnL by Desk", ...}],
  "artifacts": [{"type": "chart_html", "name": "pnl_chart.html", "path": "..."}],
  "suggestions": ["Break down Alpha by ticker?", "Compare to last quarter?"],
  "execution_trace": [
    {"node": "classify_intent", "status": "completed", "duration_ms": 450},
    {"node": "select_agents", "status": "completed", "duration_ms": 380},
    {"node": "trade_analyst", "status": "completed", "duration_ms": 520},
    ...
  ],
  "confidence": 0.92,
  "timing_ms": 3450.5
}
```

### POST /execute/stream

Same input as /execute but returns SSE events:

```
data: {"run_id": "uuid", "session_id": "uuid", "status": "started"}
data: {"step": "classify_intent", "status": "started"}
data: {"step": "classify_intent", "status": "completed", "summary": "domain=trade, intent=query_data"}
data: {"step": "select_agents", "status": "started"}
data: {"step": "select_agents", "status": "completed", "summary": "Selected: [trade_agent]"}
data: {"step": "trade_analyst", "status": "started"}
...
data: {"status": "completed", "answer": "Desk Alpha leads...", "timing_ms": 3450.5}
```

### GET /execute/{run_id}

Poll result of a previous run.

### GET /sessions/{session_id}

Get session info and conversation history.

---

## Agent Pipeline Flow

### Master Agent Flow

```
User Query
    |
    v
[classify_intent]
    |  Determines: domain, intent, desired_output, complexity
    v
[select_agents]
    |  Picks best agent(s) from registry using LLM + heuristics
    |  Strategies: sequential | parallel | pipeline
    v
[execute_agents]
    |  Invokes sub-graph(s), collects results
    |  Handles: timeouts, partial results, errors
    v
[merge_results]
    |  Synthesizes unified answer from all agents
    |  Attaches: data, charts, artifacts, suggestions
    v
Final Response
```

### Trade Agent Flow (with retry loop)

```
                +-> [trade_analyst]
                |       | domain context
                |       v
                |   [query_analyst]
                |       | structured intent
                |       v
                |   [query_planner]
                |       | execution strategy
                |       v
                |   [schema_analyzer]  <-- reads ClickHouse system tables
                |       | real schema info
                |       v
 retry (max 3)  |   [query_builder]   <-- generates ClickHouse SQL
 on validation  |       | SQL + parameters
 or exec error  |       v
                |   [query_validator] <-- programmatic + LLM validation
                |       |
                |       +-- FAIL --> retry_feedback --> [query_builder]
                |       |
                |       +-- PASS
                |       v
                |   [query_executor]  <-- executes against ClickHouse
                |       |
                |       +-- syntax error --> retry --> [query_builder]
                |       |
                |       +-- SUCCESS
                |       v
                +-- [details_analyzer]
                        | narrative, charts, exports
                        v
                    Return to Master
```

---

## Session Management & Multi-Server Scalability

### The Problem

When deploying across multiple servers behind a load balancer, you need:
- Conversation continuity (user can resume a session on any server)
- Run tracking (poll results even if the run executed on a different server)
- Graph checkpoint persistence (for long-running or interruptible flows)

### The Solution: PostgreSQL Session Layer

PostgreSQL serves as the single source of truth for all session state:

```
agent_sessions    - session lifecycle, current server, user mapping
agent_messages    - full conversation history per session
agent_runs        - run status, results, traces, artifacts
langgraph_checkpoints - graph state for resumable execution
```

### How It Works

1. **Request arrives** at any server (via load balancer)
2. Server calls `get_or_create_session(session_id)`
3. If session exists in PG (created by ANY server), it's loaded and the `server_id` is updated to the current server
4. If session doesn't exist, a new one is created
5. Run executes on the current server
6. Results are persisted to PG
7. Any subsequent request (even to a different server) can read the session, history, and results

### Server Identity

Each server has a unique `SERVER_ID` (auto-generated UUID or set via environment variable). This is tracked in the session record so you can see which server last handled a session (useful for debugging).

### Session Cleanup

A background task runs on every server at the configured interval. It marks sessions as inactive if they haven't been touched in `SESSION_TTL_HOURS`. Since all servers share the same PG, only one needs to succeed for cleanup to work. The `UPDATE` is idempotent.

### Redis Role in Multi-Server

Redis provides:
- **Schema cache**: ClickHouse schema introspection results (shared across servers)
- **Query result cache**: Short-TTL cache for repeated identical queries
- **Rate limiting state**: Request counts shared across all servers
- **Run deduplication**: Prevents the same query from executing twice within 5 seconds

### Scaling Guidelines

| Scale | Setup | Notes |
|-------|-------|-------|
| < 50 concurrent users | 1 server, 2-4 workers | Single `docker compose up` |
| 50-500 users | 3 servers + nginx | `docker compose --profile multi-server up` |
| 500-5000 users | K8s Deployment (5-20 pods) | Use PgBouncer for PG connection pooling |
| 5000+ users | K8s + read replicas | PG read replicas for session reads, Redis Cluster |

### Load Balancer Config

The included `nginx.conf` uses `ip_hash` for sticky sessions. This is preferred for SSE streaming stability but NOT required for correctness (any server can handle any request). If a server dies, the load balancer routes to another server and the session resumes seamlessly from PG.

---

## Adding New Agents

See [docs/ADDING_NEW_AGENTS.md](docs/ADDING_NEW_AGENTS.md) for a complete step-by-step guide.

**TL;DR:**
1. Create `agents/your_agent/` with `__init__.py`, `state.py`, `graph.py`, `nodes/`
2. Define state, nodes, and graph (follow the existing trade agent pattern)
3. Register via `AgentRegistry.register(descriptor, graph_factory)` in `__init__.py`
4. Add `import agents.your_agent` to `agents/__init__.py`
5. The Master Agent automatically discovers and routes to your new agent

---

## Tools & Capabilities

### Built-in Tools

| Tool | Description | Used By |
|------|-------------|---------|
| `execute_clickhouse_query` | Run read-only SQL against ClickHouse | Trade Agent |
| `get_table_schema` | Introspect ClickHouse table columns | Schema Analyzer |
| `list_clickhouse_tables` | List all tables in a database | Schema Analyzer |
| `generate_plotly_chart` | Create interactive HTML + PNG charts | Details Analyzer |
| `generate_matplotlib_chart` | Static chart fallback | Details Analyzer |
| `send_email` | SMTP email with attachments | Email Dispatcher |
| `export_to_xlsx` | Excel export | Document Builder |
| `export_to_pdf` | PDF report generation | Document Builder |
| `export_to_docx` | Word document generation | Document Builder |
| `export_to_csv` | CSV export | Document Builder |
| `export_to_json` | JSON export | Document Builder |

### MCP Tool Servers (Optional)

When `AGENT_MCP_ENABLED=true`, the platform connects to configured MCP servers and their tools become available to agents. If an MCP server provides ClickHouse tools, they automatically replace the built-in ones.

```env
AGENT_MCP_ENABLED=true
AGENT_MCP_SERVER_URLS=["http://localhost:8081/mcp"]
```

---

## Human-in-the-Loop (HITL)

The platform supports pausing agent execution at critical points and waiting for human approval, modification, or clarification before proceeding.

### Enabling HITL

Pass HITL configuration in the `preferences` field of your request:

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me PnL by desk for last month",
    "preferences": {
      "hitl": {
        "enabled": true,
        "require_sql_approval": true,
        "require_email_confirmation": true,
        "require_export_confirmation": false,
        "auto_approve_timeout": null,
        "complexity_threshold": "simple"
      }
    }
  }'
```

### HITL Flow

When HITL is enabled and an interrupt is triggered:

1. The response returns with `status: "waiting_human"` and an `interrupt_id`
2. You inspect the interrupt: `GET /interrupts/{interrupt_id}`
3. You resolve it: `POST /interrupts/{interrupt_id}/resolve`
4. The pipeline continues (or stops if rejected)

### Interrupt Types

| Type | Where | What Happens |
|------|-------|-------------|
| `approval` | Before SQL execution | Shows generated SQL. Approve, modify, or reject. |
| `confirmation` | Before sending email | Shows recipients and attachments. Confirm or cancel. |
| `clarification` | After intent classification | Agent asks user to clarify ambiguous query. |
| `modification` | Before SQL execution | User can directly edit the SQL before it runs. |
| `review` | Before export generation | Review report structure before generating. |

### Resolution Actions

```bash
# Approve
POST /interrupts/{id}/resolve
{"action": "approved"}

# Reject (stops the pipeline)
POST /interrupts/{id}/resolve
{"action": "rejected", "comment": "Wrong table"}

# Modify SQL
POST /interrupts/{id}/resolve
{"action": "modified", "modifications": {"sql": "SELECT ... LIMIT 10"}}

# Clarify
POST /interrupts/{id}/resolve
{"action": "clarified", "modifications": {"answer": "Only desk Alpha"}}
```

### Auto-Approve Timeout

Set `auto_approve_timeout` (seconds) to auto-approve if the user doesn't respond in time. Set to `null` to wait indefinitely. Clarification interrupts never auto-approve.

### Complexity Threshold

Set `complexity_threshold` to only trigger HITL for queries above a certain complexity level. Options: `simple` (always), `moderate`, `complex`. This is determined by the intent classifier.

---

## Security Model

### Defense-in-Depth for SQL Safety

Three independent layers protect against SQL injection and dangerous operations:

1. **Programmatic Blocklist** (`ClickHouseService.validate_sql_safety`)
   - Blocks DDL/DML keywords (DROP, INSERT, ALTER, etc.)
   - Requires SELECT/WITH/SHOW/DESCRIBE/EXPLAIN start
   - Blocks semicolons (multi-statement injection)
   - Requires LIMIT clause
   - Restricts system table access

2. **LLM Semantic Validation** (`query_validator` node)
   - Checks column names exist in schema
   - Validates type compatibility
   - Checks aggregation/GROUP BY consistency
   - Scores query performance

3. **ClickHouse Session-Level Readonly** (`clickhouse-connect` settings)
   - `readonly=1` prevents any write operation at the database level
   - Even if layers 1 and 2 fail, ClickHouse itself rejects writes

### API Authentication

Set `AGENT_API_KEY` to enable authentication. Provide the key via:
- `X-API-Key: your-key` header, or
- `Authorization: Bearer your-key` header

Leave `AGENT_API_KEY` empty to disable authentication (development only).

### Rate Limiting

Configurable per-endpoint rate limits via SlowAPI with Redis backend. Default: 60 requests/minute per IP. Shared across all servers via Redis.

---

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=. --cov-report=term-missing

# Run specific test categories
pytest tests/test_clickhouse.py          # SQL safety tests
pytest tests/test_master_agent.py        # Routing and selection tests
pytest tests/test_trade_agent.py         # Sub-agent node tests
pytest tests/test_session_manager.py     # PG session tests
pytest tests/test_integration.py         # End-to-end pipeline tests
```

Test architecture:
- All tests use mocked services (no real LLM/CH/PG needed)
- `conftest.py` provides reusable fixtures for mock services and sample data
- Integration tests exercise real LangGraph compilation and invocation
- SQL safety tests cover 20+ attack vectors

---

## Deployment

### Development

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production (Single Server)

```bash
gunicorn main:app \
  -k uvicorn.workers.UvicornWorker \
  -w 4 \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --graceful-timeout 30
```

### Production (Multi-Server with Docker)

```bash
docker compose --profile multi-server up -d
```

### Production (Kubernetes)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-platform
spec:
  replicas: 5
  selector:
    matchLabels:
      app: agent-platform
  template:
    spec:
      containers:
      - name: app
        image: agent-platform:latest
        ports:
        - containerPort: 8000
        env:
        - name: SERVER_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: AGENT_PG_HOST
          value: pg-service.default.svc.cluster.local
        - name: AGENT_REDIS_URL
          value: redis://redis-service:6379/0
        envFrom:
        - secretRef:
            name: agent-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
```

---

## Configuration Reference

All configuration is via environment variables prefixed with `AGENT_`. See `.env.example` for the complete list with descriptions.

| Category | Key Variables | Description |
|----------|--------------|-------------|
| LLM | `AGENT_LLM_PROVIDER`, `AGENT_LLM_MODEL`, `AGENT_LLM_API_KEY` | LLM provider and credentials |
| ClickHouse | `AGENT_CH_HOST`, `AGENT_CH_DATABASE`, `AGENT_CH_QUERY_TIMEOUT` | Analytical database |
| PostgreSQL | `AGENT_PG_HOST`, `AGENT_PG_DATABASE`, `AGENT_PG_PASSWORD` | Session store |
| Redis | `AGENT_REDIS_URL`, `AGENT_CACHE_TTL` | Cache and rate limiting |
| Security | `AGENT_API_KEY`, `AGENT_SQL_BLOCKED_KEYWORDS` | Auth and SQL safety |
| Concurrency | `AGENT_MAX_CONCURRENT_RUNS`, `AGENT_RUN_TIMEOUT` | Resource limits |
| Sessions | `AGENT_SESSION_TTL_HOURS`, `AGENT_SESSION_CLEANUP_INTERVAL_MINUTES` | Lifecycle |

---

## Project Structure

```
agent_platform/
    main.py                     # FastAPI app entrypoint
    config.py                   # Pydantic settings
    dependencies.py             # FastAPI dependency injection
    agents/
        __init__.py             # Agent imports (triggers registration)
        registry.py             # Central agent registry
        master_agent.py         # Master router graph
        trade/                  # Trade analysis agent
            __init__.py         #   Registration
            state.py            #   State TypedDict
            graph.py            #   LangGraph with retry loop
            nodes/              #   8 sub-agent nodes
        reporting/              # Report generation agent
            __init__.py         #   Full agent (plan, fetch, chart, doc, email)
        analytics/              # Trend & anomaly agent
            __init__.py         #   Full agent (trend, anomaly, summary, rank)
    tools/
        clickhouse_tools.py     # ClickHouse LangChain tools
        plotting_tools.py       # Plotly + matplotlib
        email_tools.py          # SMTP with attachments
        export_tools.py         # PDF, XLSX, DOCX, CSV, JSON
        mcp_client.py           # MCP tool server client
    services/
        clickhouse_service.py   # CH connection + SQL safety
        llm_service.py          # Multi-provider LLM abstraction
        cache_service.py        # Redis cache
        session_manager.py      # PostgreSQL session persistence
    middleware/
        auth.py                 # API key authentication
        rate_limiter.py         # SlowAPI rate limiting
    models/
        __init__.py             # All Pydantic models
    tests/
        conftest.py             # Shared fixtures
        test_clickhouse.py      # 20+ SQL safety tests
        test_master_agent.py    # Routing and selection tests
        test_trade_agent.py     # Sub-agent node tests
        test_session_manager.py # PG session tests
        test_integration.py     # End-to-end pipeline tests
    docs/
        ADDING_NEW_AGENTS.md    # Step-by-step guide
    docker-compose.yml          # Full stack with multi-server profile
    Dockerfile                  # Production image
    nginx.conf                  # Load balancer config
    requirements.txt            # Python dependencies
    pytest.ini                  # Test configuration
    .env.example                # Configuration template
```
