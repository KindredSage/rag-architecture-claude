from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from agents.registry import AgentRegistry
from middleware.auth import verify_api_key
from models import AgentInfoResponse

router = APIRouter(tags=["agents"])


@router.post("/agents", response_model=list[AgentInfoResponse])
async def list_agents(_: Any = Depends(verify_api_key)):
    """Return all registered agents with capabilities and examples."""
    return [
        AgentInfoResponse(
            agent_id=d.agent_id,
            name=d.name,
            description=d.description,
            capabilities=[c.value for c in d.capabilities],
            sub_agents=d.sub_agents,
            example_queries=d.example_queries,
            version=d.version,
            enabled=d.enabled,
        )
        for d in AgentRegistry.get_all()
    ]
