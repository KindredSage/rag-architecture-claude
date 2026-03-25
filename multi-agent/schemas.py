"""
api/models/request.py  &  api/models/response.py (combined)
------------------------------------------------------------
All FastAPI Pydantic schemas live here so they can be imported
cleanly by route handlers and the orchestration layer.
"""

# ── request.py ────────────────────────────────────────────────────────────────
from pydantic import BaseModel, Field
from typing import Any, Optional


class ExecuteRequest(BaseModel):
    """Payload sent to POST /execute."""

    query: str = Field(
        ...,
        description="Natural-language question or command",
        examples=["Give me trade details for file id 999 from My_Table"],
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional key-value context passed to the agent",
    )


# ── response.py ───────────────────────────────────────────────────────────────
class TraceStep(BaseModel):
    """Single reasoning / execution step captured during agent execution."""

    node: str = Field(..., description="Name of the LangGraph node")
    action: str = Field(..., description="What the node did")
    output: Any = Field(None, description="Node output (may be partial)")


class ExecuteResponse(BaseModel):
    """Structured response returned by POST /execute."""

    answer: str = Field(..., description="Final synthesised answer")
    agent_used: str = Field(..., description="Name of the agent that handled the query")
    steps: list[TraceStep] = Field(default_factory=list, description="Reasoning trace")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0-1]")


class AgentInfo(BaseModel):
    """Metadata for a single registered agent (returned by GET /agents)."""

    name: str
    description: str
    capabilities: list[str]
