"""
Pydantic models for API requests, responses, and internal data structures.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────


class AgentCapability(str, Enum):
    SQL_QUERY = "sql_query"
    PLOTTING = "plotting"
    REPORT_GENERATION = "report_generation"
    EMAIL = "email"
    SUMMARY = "summary"
    ANOMALY_DETECTION = "anomaly_detection"
    DATA_EXPORT = "data_export"
    SCHEMA_EXPLORATION = "schema_exploration"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_HUMAN = "waiting_human"  # paused for human-in-the-loop


class ExecutionStrategy(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"


class ArtifactType(str, Enum):
    CHART_HTML = "chart_html"
    CHART_PNG = "chart_png"
    PDF = "pdf"
    XLSX = "xlsx"
    DOCX = "docx"
    CSV = "csv"
    JSON = "json"


# ── Agent Metadata ───────────────────────────────────────────────


class AgentDescriptor(BaseModel):
    agent_id: str
    name: str
    description: str
    capabilities: list[AgentCapability]
    domain_keywords: list[str]
    sub_agents: list[str]
    example_queries: list[str] = Field(default_factory=list)
    priority: int = 0
    version: str = "1.0.0"
    enabled: bool = True


class AgentInfoResponse(BaseModel):
    agent_id: str
    name: str
    description: str
    capabilities: list[str]
    sub_agents: list[str]
    example_queries: list[str]
    version: str
    enabled: bool


# ── Execution Trace ──────────────────────────────────────────────


class TraceStep(BaseModel):
    node: str
    status: str = "started"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: float | None = None
    input_summary: str = ""
    output_summary: str = ""
    tokens_used: int | None = None
    error: str | None = None


class Artifact(BaseModel):
    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: ArtifactType
    name: str
    path: str
    size_bytes: int = 0
    mime_type: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class VisualizationConfig(BaseModel):
    chart_type: str
    title: str
    x_axis: str | None = None
    y_axis: str | None = None
    color: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


# ── API Request / Response ───────────────────────────────────────


class ExecuteRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    session_id: str | None = None
    user_id: str | None = None
    preferences: dict[str, Any] = Field(default_factory=dict)
    agent_id: str | None = None  # force a specific agent
    stream: bool = False


class ExecuteResponse(BaseModel):
    run_id: str
    session_id: str
    status: RunStatus
    answer: str = ""
    data: dict[str, Any] | None = None
    visualizations: list[VisualizationConfig] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    execution_trace: list[TraceStep] = Field(default_factory=list)
    confidence: float = 0.0
    timing_ms: float = 0.0
    error: str | None = None


class SessionInfo(BaseModel):
    session_id: str
    user_id: str | None
    created_at: datetime
    last_active: datetime
    message_count: int
    server_id: str


class HealthResponse(BaseModel):
    status: str
    version: str
    server_id: str
    clickhouse: str
    postgres: str
    redis: str
    active_runs: int
    registered_agents: int


# ── Internal State Models (used by agent graphs) ────────────────


class IntentAnalysis(BaseModel):
    primary_domain: str = "general"
    intent: str = "query_data"
    entities: list[str] = Field(default_factory=list)
    desired_output: str = "table"
    complexity: str = "simple"
    requires_multi_agent: bool = False
    sub_intents: list[str] = Field(default_factory=list)
    time_range: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    ambiguity_notes: str = ""


class AgentSelection(BaseModel):
    agent_id: str
    reason: str
    confidence: float = 1.0


class RoutingDecision(BaseModel):
    selected_agents: list[AgentSelection]
    execution_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    context_overrides: dict[str, Any] = Field(default_factory=dict)


class QueryValidationResult(BaseModel):
    is_valid: bool = False
    security_passed: bool = False
    performance_score: int = 5
    issues: list[dict[str, str]] = Field(default_factory=list)
    suggested_fixes: list[str] = Field(default_factory=list)
    approved_sql: str = ""


class QueryExecutionResult(BaseModel):
    success: bool = False
    data: list[dict[str, Any]] = Field(default_factory=list)
    columns: list[dict[str, str]] = Field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0
    bytes_read: int = 0
    truncated: bool = False
    error: str | None = None


# ── Human-in-the-Loop Models ────────────────────────────────────


class InterruptType(str, Enum):
    """Types of human intervention points."""
    APPROVAL = "approval"           # approve/reject before proceeding
    CLARIFICATION = "clarification" # agent needs more info from user
    MODIFICATION = "modification"   # user can edit generated content (e.g. SQL)
    CONFIRMATION = "confirmation"   # confirm before irreversible action (email)
    REVIEW = "review"               # review results before finalizing


class InterruptStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    CLARIFIED = "clarified"
    EXPIRED = "expired"
    AUTO_APPROVED = "auto_approved"


class InterruptRequest(BaseModel):
    """Created by an agent node when it needs human input."""
    interrupt_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    session_id: str
    interrupt_type: InterruptType
    node_name: str                  # which node paused
    agent_id: str                   # which agent
    title: str                      # human-readable title
    description: str                # what the agent is asking
    payload: dict[str, Any] = Field(default_factory=dict)
    # payload examples:
    #   APPROVAL:      {"sql": "SELECT ...", "estimated_rows": 50000}
    #   CLARIFICATION: {"question": "Which desk?", "options": ["Alpha", "Beta"]}
    #   MODIFICATION:  {"sql": "SELECT ...", "editable_fields": ["sql"]}
    #   CONFIRMATION:  {"action": "send_email", "to": ["a@b.com"], "subject": "..."}
    auto_approve_seconds: int | None = None  # auto-approve after N seconds (None = wait forever)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class InterruptResponse(BaseModel):
    """Submitted by the user to resolve an interrupt."""
    interrupt_id: str
    action: InterruptStatus         # approved / rejected / modified / clarified
    modifications: dict[str, Any] = Field(default_factory=dict)
    # modifications examples:
    #   MODIFIED:  {"sql": "SELECT desk, SUM(pnl) ... LIMIT 50"}
    #   CLARIFIED: {"answer": "Alpha desk only"}
    comment: str = ""               # optional user comment


class InterruptInfo(BaseModel):
    """API response model for interrupt details."""
    interrupt_id: str
    run_id: str
    session_id: str
    interrupt_type: InterruptType
    status: InterruptStatus
    node_name: str
    agent_id: str
    title: str
    description: str
    payload: dict[str, Any]
    auto_approve_seconds: int | None
    created_at: datetime
    resolved_at: datetime | None = None
    resolution: dict[str, Any] | None = None


class HITLConfig(BaseModel):
    """Per-request HITL configuration. Passed in ExecuteRequest.preferences."""
    enabled: bool = False
    require_sql_approval: bool = True       # pause before executing SQL
    require_email_confirmation: bool = True  # pause before sending email
    require_export_confirmation: bool = False # pause before generating exports
    auto_approve_timeout: int | None = None  # seconds, None = wait forever
    # Complexity threshold: only interrupt if query complexity >= this level
    complexity_threshold: str = "simple"     # simple | moderate | complex
