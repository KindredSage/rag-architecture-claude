#!/usr/bin/env python3
"""
examples/api_calls.py
---------------------
Demonstrates every public API endpoint.
Run after starting the server with:   uvicorn app.main:app --reload

Requires: pip install httpx
"""

import asyncio
import json
import httpx

BASE_URL = "http://localhost:8000"


async def example_health():
    print("\n" + "═" * 60)
    print("GET /health")
    print("═" * 60)
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_URL}/health")
    print(json.dumps(r.json(), indent=2))


async def example_list_agents():
    print("\n" + "═" * 60)
    print("GET /agents  –  List registered agents")
    print("═" * 60)
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_URL}/agents")
    for agent in r.json():
        print(f"\n  🤖  {agent['name']}")
        print(f"      {agent['description']}")
        print(f"      Capabilities: {', '.join(agent['capabilities'][:3])}...")


async def example_execute_trade_query():
    """
    Primary example: retrieve trade details for file id 999.
    This exercises the full Master → TradeAgent pipeline.
    """
    print("\n" + "═" * 60)
    print("POST /execute  –  Trade data query (full pipeline)")
    print("═" * 60)

    payload = {
        "query": "Give me trade details for file id 999 from My_Table",
        "context": {
            "user_id": "analyst_01",
            "department": "trading_desk",
        }
    }
    print(f"\nRequest payload:\n{json.dumps(payload, indent=2)}")

    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{BASE_URL}/execute", json=payload)

    resp = r.json()
    print(f"\n{'─'*60}")
    print(f"Response (HTTP {r.status_code}):")
    print(f"{'─'*60}")
    print(f"  agent_used : {resp.get('agent_used')}")
    print(f"  confidence : {resp.get('confidence'):.2f}")
    print(f"  steps      : {len(resp.get('steps', []))} nodes traced")
    print(f"\n  answer:\n")
    print("  " + resp.get("answer", "").replace("\n", "\n  "))
    print(f"\n  trace:")
    for step in resp.get("steps", []):
        print(f"    [{step['node']}] {step['action']}")


async def example_execute_direct_answer():
    """Example of a general query that goes to the Direct Answer path."""
    print("\n" + "═" * 60)
    print("POST /execute  –  General question (direct answer path)")
    print("═" * 60)

    payload = {
        "query": "What is an FX Forward trade?",
        "context": {}
    }
    print(f"\nRequest payload:\n{json.dumps(payload, indent=2)}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{BASE_URL}/execute", json=payload)

    resp = r.json()
    print(f"\n  agent_used : {resp.get('agent_used')}")
    print(f"  confidence : {resp.get('confidence'):.2f}")
    print(f"\n  answer:\n")
    print("  " + resp.get("answer", "").replace("\n", "\n  "))


async def example_curl_commands():
    """Print equivalent curl commands for documentation."""
    print("\n" + "═" * 60)
    print("Equivalent curl commands")
    print("═" * 60)

    print("""
# 1. Health check
curl -s http://localhost:8000/health | python -m json.tool

# 2. List agents
curl -s http://localhost:8000/agents | python -m json.tool

# 3. Execute trade query
curl -s -X POST http://localhost:8000/execute \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "Give me trade details for file id 999 from My_Table",
    "context": {"user_id": "analyst_01"}
  }' | python -m json.tool

# 4. Execute general query (direct answer)
curl -s -X POST http://localhost:8000/execute \\
  -H "Content-Type: application/json" \\
  -d '{"query": "What is an FX Forward trade?", "context": {}}' \\
  | python -m json.tool
""")


async def main():
    await example_health()
    await example_list_agents()
    await example_execute_trade_query()
    await example_execute_direct_answer()
    await example_curl_commands()


if __name__ == "__main__":
    asyncio.run(main())
