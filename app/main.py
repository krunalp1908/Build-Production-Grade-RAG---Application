# ============================================================
# CRITICAL:
# logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================

import logfire
import os

from dotenv import load_dotenv

import uuid


load_dotenv()

logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN")
)


# ============================================================
# Now safe to import application modules
# ============================================================

from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import Optional

from app.agents.graph import rag_agent
from app.guardrails import initialize_rails, guard


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Enterprise Agentic RAG API"
)


# ============================================================
# Startup
# ============================================================

@app.on_event("startup")
def startup_event():
    initialize_rails()


# ============================================================
# Request model
# ============================================================

class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"


# ============================================================
# Health check
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Enterprise LangGraph RAG API is live."
    }


# ============================================================
# Graph visualization
# ============================================================

@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """

    try:
        png_bytes = (
            rag_agent
            .get_graph()
            .draw_mermaid_png()
        )

        return Response(
            content=png_bytes,
            media_type="image/png",
        )

    except Exception as e:
        return {
            "error": f"Could not generate graph image: {e}"
        }


# ============================================================
# Query endpoint
# ============================================================

@app.post("/query")
def query(request: QueryRequest):
    """
    Execute the LangGraph RAG flow with session memory.

    Guardrails are evaluated BEFORE the RAG graph.

    The guard receives the previous messages from the same
    thread so that contextual RAG follow-ups can be recognized.
    """

    q = (request.q or "").strip()

    thread_id = (
        request.thread_id
        or "default_user"
    )

    # --------------------------------------------------------
    # Configuration for LangGraph memory
    # --------------------------------------------------------

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # --------------------------------------------------------
    # Retrieve existing session state
    # --------------------------------------------------------

    try:
        state_snapshot = rag_agent.get_state(config)

        existing_state = (
            state_snapshot.values
            if state_snapshot
            else {}
        )

    except Exception as exc:
        logfire.error(
            f"❌ Could not retrieve graph state: {exc}"
        )

        existing_state = {}


    # --------------------------------------------------------
    # Existing conversation
    # --------------------------------------------------------

    prior_messages = (
        existing_state.get("messages", [])
        if isinstance(existing_state, dict)
        else []
    )


    # ========================================================
    # GATE 1 — STRICT SESSION-AWARE GUARDRAILS
    # ========================================================

    try:

        rail_fired, rail_response = guard(
            message=q,
            prior_messages=prior_messages,
        )

    except Exception as exc:

        # FAIL CLOSED
        logfire.error(
            f"❌ Guardrail execution failed: {exc}"
        )

        return {
            "question": q,
            "answer": (
                "I’m sorry, but I couldn’t safely process "
                "that request right now."
            ),
            "thought_process": [
                "Intent: Guardrail execution failure",
                "Retrieval: Skipped",
            ],
            "status": "Blocked by guardrails.",
            "sources": [],
        }


    # --------------------------------------------------------
    # Guardrail fired
    # --------------------------------------------------------

    if rail_fired:

        logfire.info(
            f"🛡️ Request handled by guardrails | "
            f"thread={thread_id}"
        )

        return {
            "question": q,
            "answer": rail_response,
            "thought_process": [
                "Intent: Guardrails Fired",
                "Retrieval: Skipped",
            ],
            "status": "Blocked by guardrails.",
            "sources": [],
        }


    # ========================================================
    # GATE 2 — LANGGRAPH RAG PIPELINE
    # ========================================================

    # Only allowed messages reach this point.
    #
    # IMPORTANT:
    # Do not put blocked messages into the RAG conversation state.
    # This prevents off-topic conversations from contaminating
    # future contextual follow-ups.

    initial_state = {
        "messages": (
            prior_messages
            + [
                {
                    "role": "user",
                    "content": q,
                }
            ]
        ),

        "current_query": q,

        "documents": (
            existing_state.get("documents", [])
            if isinstance(existing_state, dict)
            else []
        ),

        "plan": (
            existing_state.get("plan", ["Start"])
            if isinstance(existing_state, dict)
            else ["Start"]
        ),

        "status": "Initializing Graph...",
    }


    # --------------------------------------------------------
    # Execute RAG
    # --------------------------------------------------------

    try:

        final_output = rag_agent.invoke(
            initial_state,
            config=config,
        )


        return {
            "question": q,
            "answer": final_output.get(
                "final_answer"
            ),
            "thought_process": final_output.get(
                "plan"
            ),
            "status": final_output.get(
                "status"
            ),
            "sources": final_output.get(
                "documents",
                []
            ),
        }


    except Exception as e:

        logfire.error(
            f"❌ Backend Execution Failed: {e}"
        )

        return {
            "question": q,
            "answer": (
                "I apologize, but I encountered an "
                "internal error while processing your request. "
                "Please try again later."
            ),
            "thought_process": [
                "Error encountered during execution."
            ],
            "status": "error",
            "sources": [],
        }