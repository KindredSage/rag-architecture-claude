"""
chat_pipeline.py
────────────────────────────────────────────────────────────────────────────────
Production-grade scatter-gather RAG + DB agent pipeline.

Covers everything discussed:
  • ChatOpenAI pointed at OSS 120B endpoint
  • Binary + uncertain router with known-app cache
  • Clarification state machine (uncertain → ask user → re-route on original Q)
  • Yes/No classifier (handles "yep", "sure", "not really" etc.)
  • RAG fetch (KNN via vector store)
  • DB agent fetch via LangChain 1.2+ create_agent (no AgentExecutor)
  • Tools: search_applications, get_app_details, get_upstreams,
           get_downstreams, search_documentation
  • Parallel tool call note for OSS models
  • Scatter-gather: RAG + DB run concurrently, one failure won't kill other
  • Synthesizer combining both sources with contradiction detection
  • ChatSession with history trimming and full state management

Requirements:
    pip install langchain>=1.2.0 langchain-openai langchain-core pydantic

LangChain 1.2 notes:
    - create_agent replaces create_openai_tools_agent / AgentExecutor
    - Input schema is {"messages": [...]} not {"input": ..., "chat_history": [...]}
    - Plain dicts {"role": ..., "content": ...} work, no need for HumanMessage etc.
────────────────────────────────────────────────────────────────────────────────
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)


# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIG  — edit these
# ─────────────────────────────────────────────────────────────────────────────

OSS_BASE_URL  = "http://your-internal-host/v1"
OSS_API_KEY   = "your-api-key"
OSS_MODEL     = "your-oss-120b-model-name"

APP_CACHE_TTL = 300   # seconds before re-fetching app name list from DB
MAX_HISTORY   = 10    # conversation pairs kept in memory per session


# ─────────────────────────────────────────────────────────────────────────────
# 1. LLM INSTANCES
#    Three separate instances so you can swap models per role independently.
# ─────────────────────────────────────────────────────────────────────────────

# Cheap + fast — used only for routing and yes/no classification
router_llm = ChatOpenAI(
    model=OSS_MODEL,
    base_url=OSS_BASE_URL,
    api_key=OSS_API_KEY,
    temperature=0,
    max_tokens=10,          # only needs one word output
)

# Main reasoning model — used by the DB agent for tool orchestration
agent_llm = ChatOpenAI(
    model=OSS_MODEL,
    base_url=OSS_BASE_URL,
    api_key=OSS_API_KEY,
    temperature=0,
)

# Synthesis model — slightly creative, combines RAG + DB into final answer
synth_llm = ChatOpenAI(
    model=OSS_MODEL,
    base_url=OSS_BASE_URL,
    api_key=OSS_API_KEY,
    temperature=0.2,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DB LAYER
#    Replace these stubs with your actual async DB calls (SQLAlchemy, asyncpg,
#    or whatever ORM sits in front of your Oracle / Postgres / etc.)
# ─────────────────────────────────────────────────────────────────────────────

class DB:

    @staticmethod
    async def fetch_all_app_names() -> list[str]:
        """SELECT app_name FROM applications"""
        # ← replace with real DB call
        return ["odin-payments", "star-settlements", "ares-risk", "hermes-reporting"]

    @staticmethod
    async def search_apps(query: str) -> list[dict]:
        """
        Fuzzy search by app name.
        Recommended: pg_trgm similarity or Levenshtein in SQL — not in Python.
        SELECT app_id, app_name, team
        FROM   applications
        WHERE  app_name ILIKE %query%
        ORDER  BY similarity(app_name, :query) DESC
        LIMIT  5
        """
        # ← replace with real DB call
        return [{"app_id": "app-001", "app_name": "odin-payments", "team": "trade-processing"}]

    @staticmethod
    async def get_app_details(app_id: str) -> dict:
        """Full row from applications table for a confirmed app_id."""
        # ← replace with real DB call
        return {
            "app_id":       app_id,
            "app_name":     "odin-payments",
            "app_manager":  "john.doe@company.com",
            "sla":          "99.9%",
            "team":         "trade-processing",
            "environment":  "production",
        }

    @staticmethod
    async def get_upstreams(app_id: str) -> list[dict]:
        """SELECT * FROM upstreams WHERE app_id = :app_id"""
        # ← replace with real DB call
        return [
            {"upstream_name": "kafka-trade-feed", "connection_type": "kafka"},
            {"upstream_name": "oracle-trade-db",  "connection_type": "jdbc"},
        ]

    @staticmethod
    async def get_downstreams(app_id: str) -> list[dict]:
        """SELECT * FROM downstreams WHERE app_id = :app_id"""
        # ← replace with real DB call
        return [
            {"downstream_name": "star-settlements", "connection_type": "kafka"},
            {"downstream_name": "hermes-reporting", "connection_type": "rest"},
        ]


db = DB()


# ─────────────────────────────────────────────────────────────────────────────
# 3. VECTOR STORE
#    Replace with your actual store: OpenSearch / BGEM3 / Chroma / FAISS etc.
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:

    async def asimilarity_search(self, query: str, k: int = 4) -> list:
        """
        Returns list of Document(page_content=str, metadata={"source": str})
        ← replace with real KNN call to your BGEM3 + OpenSearch pipeline
        """
        return []


vector_store = VectorStore()


# ─────────────────────────────────────────────────────────────────────────────
# 4. APP NAME CACHE
#    Avoids hitting DB on every single message just for the router.
#    TTL-based — refreshes in background on first stale request.
# ─────────────────────────────────────────────────────────────────────────────

_app_cache: tuple[list[str], float] = ([], 0.0)


async def get_known_apps() -> list[str]:
    global _app_cache
    names, ts = _app_cache
    if time.time() - ts > APP_CACHE_TTL:
        try:
            names = await db.fetch_all_app_names()
            _app_cache = (names, time.time())
            logger.info(f"App cache refreshed — {len(names)} apps loaded")
        except Exception as e:
            logger.error(f"Failed to refresh app cache: {e}")
            # Return stale data rather than failing the whole request
    return names


# ─────────────────────────────────────────────────────────────────────────────
# 5. TOOLS
#    Tool docstrings are instructions to the LLM — write them carefully.
#    The LLM reads them to decide WHEN and HOW to call each tool.
#
#    OSS 120B note on parallel tool calls:
#    GPT-4o natively emits multiple tool calls in one turn for independent
#    fetches (e.g. get_upstreams + get_downstreams after resolving app_id).
#    OSS models vary — if yours serializes, the system prompt nudge below
#    helps. If it still serializes, the scatter-gather in Python (section 11)
#    compensates at the RAG vs DB level.
# ─────────────────────────────────────────────────────────────────────────────

class SearchAppsInput(BaseModel):
    query: str = Field(description="App name or partial name to search for")


@tool(args_schema=SearchAppsInput)
async def search_applications(query: str) -> dict:
    """
    Search for applications by name. ALWAYS call this first before any
    other app-specific tool to resolve the exact app name and app_id.

    Behaviour based on results:
    - 0 matches  → app does not exist. Tell the user and suggest they check spelling.
    - 1 match    → proceed with returned app_id.
    - 2+ matches → list all matches and ask the user to clarify which one.

    Never guess or proceed without a confirmed app_id.
    """
    results = await db.search_apps(query)
    if not results:
        return {
            "found": False,
            "matches": [],
            "message": f"No application found matching '{query}'. Ask user to verify the name.",
        }
    return {
        "found": True,
        "exact_match": len(results) == 1,
        "matches": results,  # each has app_id, app_name, team
    }


class AppIdInput(BaseModel):
    app_id: str = Field(description="Exact app_id as returned by search_applications")


@tool(args_schema=AppIdInput)
async def get_app_details(app_id: str) -> dict:
    """
    Fetch full details for an application: manager, SLA, team, environment, config.
    Only call after confirming exact app_id via search_applications.
    """
    return await db.get_app_details(app_id)


@tool(args_schema=AppIdInput)
async def get_upstreams(app_id: str) -> dict:
    """
    Get all upstream systems that send data TO this application.
    Only call after confirming exact app_id via search_applications.
    Returns upstream names, count, and connection types (kafka, jdbc, rest, etc.)
    If question also asks about downstreams, call get_downstreams in the same step.
    """
    rows = await db.get_upstreams(app_id)
    return {"app_id": app_id, "count": len(rows), "upstreams": rows}


@tool(args_schema=AppIdInput)
async def get_downstreams(app_id: str) -> dict:
    """
    Get all downstream systems that receive data FROM this application.
    Only call after confirming exact app_id via search_applications.
    Returns downstream names, count, and connection types (kafka, jdbc, rest, etc.)
    If question also asks about upstreams, call get_upstreams in the same step.
    """
    rows = await db.get_downstreams(app_id)
    return {"app_id": app_id, "count": len(rows), "downstreams": rows}


class SearchDocsInput(BaseModel):
    query: str = Field(description="Natural language query to search documentation")


@tool(args_schema=SearchDocsInput)
async def search_documentation(query: str) -> str:
    """
    Search internal documentation: architecture guidelines, runbooks, policies,
    integration patterns, communication protocols, app descriptions.

    Use for:
    - HOW systems work or communicate
    - Policies and standards
    - Architecture decisions
    - App purpose / what it does

    Do NOT use for specific app topology, ownership, or SLA — use DB tools for those.
    """
    docs = await vector_store.asimilarity_search(query, k=4)
    if not docs:
        return "No relevant documentation found for this query."
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


ALL_TOOLS = [
    search_applications,
    get_app_details,
    get_upstreams,
    get_downstreams,
    search_documentation,
]


# ─────────────────────────────────────────────────────────────────────────────
# 6. DB AGENT  (LangChain 1.2+ — create_agent replaces AgentExecutor)
#
#    Input  : {"messages": [...]}
#    Output : {"messages": [...]}  — last message is the agent's final output
#
#    The agent handles sequential tool calls internally:
#      Turn 1: search_applications → resolves app_id
#      Turn 2: get_upstreams + get_downstreams (parallel if model supports it)
#      Turn 3: synthesizes raw data into structured output
# ─────────────────────────────────────────────────────────────────────────────

_DB_AGENT_SYSTEM = """You are a data retrieval agent for an internal application registry.
Your only job is to fetch data using the tools provided. Do not answer questions yourself.

Rules:
1. ALWAYS call search_applications first to resolve any app name to an exact app_id.
2. If search returns 0 results → return a clear "not found" message. Do not guess.
3. If search returns multiple results → return all candidates. Do not pick one.
4. Once you have a confirmed app_id:
   - If question needs both upstreams AND downstreams → call both tools in the same step.
   - For details (manager, SLA, team) → call get_app_details.
   - For architecture / protocols → call search_documentation.
5. Return structured raw data only. No narrative, no opinions.
6. Never fabricate or infer data not returned by tools.
"""

db_agent = create_agent(
    model=agent_llm,
    tools=ALL_TOOLS,
    system_prompt=_DB_AGENT_SYSTEM,
)


# ─────────────────────────────────────────────────────────────────────────────
# 7. ROUTER
#    Three labels: general | internal | uncertain
#    Injects known app names so "odin" isn't classified as general.
#    Conservative — when in doubt returns "uncertain" and asks user.
# ─────────────────────────────────────────────────────────────────────────────

async def route(question: str) -> str:
    """Returns: 'general' | 'internal' | 'uncertain'"""
    known_apps = await get_known_apps()
    app_list   = ", ".join(known_apps) if known_apps else "none loaded"

    prompt = f"""You are a classifier for an internal platform assistant.

Known internal applications: {app_list}

Classify the question into exactly one label:

  general   — clearly generic knowledge, no connection to internal systems
  internal  — mentions or implies a known app, its topology, ownership,
               policies, architecture, or any internal system concept
  uncertain — ambiguous; could be either

Rules:
  • If ANY word in the question resembles a known app name → internal
  • Questions about what an app IS count as internal (answered via docs)
  • When in doubt → uncertain. Never guess general.

Reply with ONLY the label. No punctuation, no explanation.

Question: {question}
"""
    resp  = await router_llm.ainvoke(prompt)
    label = resp.content.strip().lower().split()[0]   # take first word only
    if label not in ("general", "internal", "uncertain"):
        logger.warning(f"Router returned unexpected label {label!r}, defaulting to uncertain")
        label = "uncertain"
    logger.info(f"Route → {label!r}  |  question: {question!r}")
    return label


# ─────────────────────────────────────────────────────────────────────────────
# 8. YES/NO CLASSIFIER
#    Handles natural language confirmations: "yep", "sure", "no not really"
# ─────────────────────────────────────────────────────────────────────────────

async def classify_yes_no(message: str) -> bool:
    prompt = f"""Does this message mean YES or NO?
Reply with ONLY: YES  or  NO

Message: {message}
"""
    resp = await router_llm.ainvoke(prompt)
    return resp.content.strip().upper().startswith("Y")


# ─────────────────────────────────────────────────────────────────────────────
# 9. RAG FETCH
#    Fast path — KNN search, no LLM involved here.
#    Returns None on empty results or errors (scatter-gather handles gracefully).
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_rag(question: str) -> Optional[str]:
    try:
        docs = await vector_store.asimilarity_search(question, k=4)
        if not docs:
            return None
        return "\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in docs
        )
    except Exception as e:
        logger.warning(f"RAG fetch failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 10. DB AGENT FETCH
#     Wraps db_agent.ainvoke with error isolation.
#     History is passed so agent is aware of conversation context
#     (e.g. "the second one" after a clarification).
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_db(question: str, history: list[dict]) -> Optional[str]:
    try:
        result = await db_agent.ainvoke({
            "messages": [
                *history,
                {"role": "user", "content": question},
            ]
        })
        # Last message in the response is the agent's final output
        return result["messages"][-1].content
    except Exception as e:
        logger.error(f"DB agent fetch failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 11. SYNTHESIZER
#     Receives both RAG and DB contexts (either can be None).
#     Explicit instructions for partial context and contradiction handling.
# ─────────────────────────────────────────────────────────────────────────────

_SYNTH_SYSTEM = """You are a helpful internal platform assistant.
Answer the user's question using ONLY the context provided below.

Rules:
  • Use whichever context section is relevant. Ignore irrelevant parts.
  • If Documentation and Application Registry contradict each other, flag it explicitly.
  • If context is partial, say what you know and what you could not find.
  • If no context was retrieved at all, say so — do not hallucinate answers.
  • Be concise and direct. No filler phrases.
"""

async def synthesize(
    question:    str,
    history:     list[dict],
    rag_context: Optional[str],
    db_context:  Optional[str],
) -> str:
    context_block = ""
    if rag_context:
        context_block += f"=== Documentation ===\n{rag_context}\n\n"
    if db_context:
        context_block += f"=== Application Registry ===\n{db_context}\n\n"
    if not context_block:
        context_block = "No relevant context was retrieved from any source."

    messages = [
        {"role": "system",    "content": _SYNTH_SYSTEM},
        *history,
        {"role": "user",      "content": f"{context_block}\nQuestion: {question}"},
    ]
    resp = await synth_llm.ainvoke(messages)
    return resp.content


# ─────────────────────────────────────────────────────────────────────────────
# 12. PIPELINE RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    answer:      str
    route:       str
    rag_context: Optional[str] = None
    db_context:  Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# 13. EXECUTE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

async def _execute_internal(question: str, history: list[dict]) -> PipelineResult:
    """
    Scatter-gather:
      RAG fetch  ──┐
                   ├── asyncio.gather ──► synthesize ──► answer
      DB agent   ──┘

    return_exceptions=True ensures one source failing doesn't kill the other.
    """
    rag_result, db_result = await asyncio.gather(
        fetch_rag(question),
        fetch_db(question, history),
        return_exceptions=True,
    )

    if isinstance(rag_result, Exception):
        logger.error(f"RAG scatter error: {rag_result}")
        rag_result = None
    if isinstance(db_result, Exception):
        logger.error(f"DB scatter error: {db_result}")
        db_result = None

    answer = await synthesize(question, history, rag_result, db_result)
    return PipelineResult(
        answer=answer,
        route="internal",
        rag_context=rag_result,
        db_context=db_result,
    )


async def _execute_general(question: str, history: list[dict]) -> PipelineResult:
    """Fast path — no retrieval, direct LLM answer."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        *history,
        {"role": "user",   "content": question},
    ]
    resp = await synth_llm.ainvoke(messages)
    return PipelineResult(answer=resp.content, route="general")


# ─────────────────────────────────────────────────────────────────────────────
# 14. CHAT SESSION
#
#    State machine:
#
#      IDLE
#       │
#       │  route == "uncertain"
#       ▼
#    AWAITING_CONFIRMATION  ──► user says yes/no
#       │
#       │  re-route original question (not "yes")
#       ▼
#      IDLE
#
#    The critical detail: pending_question stores the ORIGINAL question.
#    The confirmation reply ("yes"/"no") is NEVER passed to RAG or DB.
# ─────────────────────────────────────────────────────────────────────────────

class SessionState(Enum):
    IDLE                  = "idle"
    AWAITING_CONFIRMATION = "awaiting_confirmation"


class ChatSession:
    """
    One instance per user session / conversation.

    Usage:
        session = ChatSession()
        reply   = await session.chat("What is odin?")
        reply   = await session.chat("yes")           # confirmation if asked
        reply   = await session.chat("who owns it?")  # follows up with history
    """

    _CLARIFICATION = (
        "Just to confirm — is your question about an internal application or system? (yes / no)"
    )

    def __init__(self, max_history: int = MAX_HISTORY):
        self.history:          list[dict]    = []
        self.state:            SessionState  = SessionState.IDLE
        self.pending_question: Optional[str] = None
        self.max_history:      int           = max_history

    # ── Public entrypoint ────────────────────────────────────────────────────

    async def chat(self, user_message: str) -> str:
        if self.state == SessionState.AWAITING_CONFIRMATION:
            return await self._handle_confirmation(user_message)
        return await self._handle_question(user_message)

    # ── Normal question flow ─────────────────────────────────────────────────

    async def _handle_question(self, question: str) -> str:
        route_label = await route(question)

        if route_label == "uncertain":
            self.pending_question = question
            self.state = SessionState.AWAITING_CONFIRMATION
            # Record clarification exchange in history
            self._append(question, self._CLARIFICATION)
            return self._CLARIFICATION

        result = await (
            _execute_internal(question, self.history)
            if route_label == "internal"
            else _execute_general(question, self.history)
        )
        self._append(question, result.answer)
        return result.answer

    # ── Confirmation flow ─────────────────────────────────────────────────────

    async def _handle_confirmation(self, user_message: str) -> str:
        original = self.pending_question

        # Reset BEFORE awaiting — prevents state corruption on concurrent calls
        self.state            = SessionState.IDLE
        self.pending_question = None

        is_yes = await classify_yes_no(user_message)
        result = await (
            _execute_internal(original, self.history)
            if is_yes
            else _execute_general(original, self.history)
        )

        # Append user confirmation + final answer (the clarification pair is already in history)
        self._append(user_message, result.answer)
        return result.answer

    # ── History management ────────────────────────────────────────────────────

    def _append(self, question: str, answer: str) -> None:
        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant",  "content": answer})
        # Trim to last N pairs — prevents context window bloat over long sessions
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]

    def reset(self) -> None:
        """Hard reset — call when starting a genuinely new topic."""
        self.history.clear()
        self.state            = SessionState.IDLE
        self.pending_question = None
        logger.info("ChatSession reset")


# ─────────────────────────────────────────────────────────────────────────────
# 15. EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    session = ChatSession()

    test_questions = [
        # ── Uncertain — triggers clarification ──────────────────────────────
        "What is odin?",
        "yes",                                            # confirms → internal → RAG + DB

        # ── DB only ─────────────────────────────────────────────────────────
        "What upstream systems send trade data to odin?",
        "Which systems are downstreams of odin?",
        "Who is the app manager of odin?",

        # ── Hybrid (RAG + DB scatter-gather) ────────────────────────────────
        "How does odin communicate with star?",

        # ── Both upstreams + downstreams in one question ─────────────────────
        # Agent should call get_upstreams + get_downstreams in parallel after
        # resolving app_id — if OSS model supports parallel tool calls.
        "List the number of upstreams and downstreams for odin",

        # ── General fast path ────────────────────────────────────────────────
        "What is a Kafka consumer group?",

        # ── Typo — tests not-found path ──────────────────────────────────────
        "What are the upstreams for app odn?",

        # ── Follow-up using history ──────────────────────────────────────────
        "Who manages that app?",                          # refers to odin from history
    ]

    for q in test_questions:
        print(f"\n{'─' * 64}")
        print(f"  User : {q}")
        reply = await session.chat(q)
        print(f"  Agent: {reply}")

    print(f"\n{'─' * 64}")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())