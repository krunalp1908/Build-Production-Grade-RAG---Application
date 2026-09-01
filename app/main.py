# ============================================================
# ENVIRONMENT
# ============================================================

import os
import uuid

from dotenv import load_dotenv


load_dotenv()


# ============================================================
# LOGFIRE
# ============================================================

import logfire


logfire.configure(
    token=os.getenv(
        "LOGFIRE_TOKEN"
    )
)


# ============================================================
# FASTAPI
# ============================================================

from fastapi import (
    FastAPI,
    Response,
)

from pydantic import BaseModel

from typing import Optional


# ============================================================
# APPLICATION MODULES
# ============================================================

from app.agents.graph import (
    rag_agent,
)

from app.guardrails import (
    initialize_rails,
    guard,
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Enterprise Agentic RAG API",
    description=(
        "Documentation-grounded Agentic RAG "
        "Assistant with session memory and "
        "strict guardrails."
    ),
    version="1.0.0",
)


# ============================================================
# STARTUP
# ============================================================

@app.on_event(
    "startup"
)
def startup_event():

    initialize_rails()

    logfire.info(
        "🚀 Enterprise Agentic RAG API started."
    )


# ============================================================
# REQUEST MODEL
# ============================================================

class QueryRequest(
    BaseModel
):

    q: str

    thread_id: Optional[str] = None


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": (
            "Enterprise LangGraph RAG API is live."
        )
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# GRAPH VISUALIZATION
# ============================================================

@app.get("/graph")
def get_graph_image():

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

    except Exception as exc:

        logfire.error(
            "❌ Could not generate graph image: "
            f"{exc}"
        )

        return {
            "error": (
                f"Could not generate graph image: {exc}"
            )
        }


# ============================================================
# QUERY ENDPOINT
# ============================================================

@app.post("/query")
def query(
    request: QueryRequest,
):

    # ========================================================
    # CLEAN INPUT
    # ========================================================

    q = (
        request.q or ""
    ).strip()


    # ========================================================
    # SESSION ID
    # ========================================================

    thread_id = (
        request.thread_id
        or str(uuid.uuid4())
    )


    logfire.info(
        "💬 Incoming request | "
        f"thread={thread_id}"
    )


    # ========================================================
    # EMPTY REQUEST
    # ========================================================

    if not q:

        return {
            "question": q,

            "answer": (
                "Please enter a question."
            ),

            "thought_process": [
                "Guardrail: Empty request blocked",
                "Retrieval: Skipped",
                "LLM synthesis: Skipped",
            ],

            "status": (
                "No query provided."
            ),

            "sources": [],

            "thread_id": thread_id,
        }


    # ========================================================
    # LANGGRAPH CONFIG
    # ========================================================

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }


    # ========================================================
    # READ EXISTING MEMORY
    # ========================================================
    #
    # This is ONLY used to give the guardrail enough context
    # to understand conversation-memory requests.
    #
    # We DO NOT manually inject these messages into the new
    # graph state.
    #
    # LangGraph's checkpointer handles that.
    #
    # ========================================================

    try:

        snapshot = (
            rag_agent.get_state(
                config
            )
        )

        existing_state = (
            snapshot.values
            if snapshot
            else {}
        )

    except Exception as exc:

        logfire.warning(
            "⚠️ Could not read existing session "
            f"memory: {exc}"
        )

        existing_state = {}


    # ========================================================
    # PREVIOUS MESSAGES
    # ========================================================

    prior_messages = []

    if isinstance(
        existing_state,
        dict,
    ):

        prior_messages = (
            existing_state.get(
                "messages",
                [],
            )
        )


    # ========================================================
    # GUARDRAIL
    # ========================================================

    try:

        (
            rail_fired,
            rail_response,
            classification,
        ) = guard(
            message=q,
            prior_messages=prior_messages,
        )

    except Exception as exc:

        # ----------------------------------------------------
        # FAIL CLOSED
        # ----------------------------------------------------

        logfire.error(
            "❌ Guardrail execution failed: "
            f"{exc}"
        )

        return {
            "question": q,

            "answer": (
                "I couldn't safely process "
                "that request."
            ),

            "thought_process": [
                "Guardrail: Failed closed",
                "Retrieval: Skipped",
                "LLM synthesis: Skipped",
            ],

            "status": (
                "Blocked by safety guardrail."
            ),

            "sources": [],

            "thread_id": thread_id,
        }


    # ========================================================
    # GUARDRAIL-HANDLED REQUEST
    # ========================================================

    if rail_fired:

        logfire.info(
            "🛡️ Request handled by guardrail | "
            f"classification={classification} | "
            f"thread={thread_id}"
        )

        return {
            "question": q,

            "answer": rail_response,

            "thought_process": [
                (
                    "Guardrail: "
                    f"{classification}"
                ),
                "Retrieval: Skipped",
                "LLM synthesis: Skipped",
            ],

            "status": (
                "Handled by guardrails."
            ),

            "sources": [],

            "thread_id": thread_id,
        }


    # ========================================================
    # INITIAL GRAPH STATE
    # ========================================================
    #
    # IMPORTANT:
    #
    # Only the NEW user message is supplied.
    #
    # Previous conversation is restored automatically by
    # MemorySaver using thread_id.
    #
    # ========================================================

    initial_state = {

        "messages": [
            {
                "role": "user",
                "content": q,
            }
        ],

        "current_query": "",

        "intent": classification,

        "documents": [],

        "plan": [
            (
                "Guardrail: "
                f"{classification}"
            )
        ],

        "status": (
            "Initializing LangGraph..."
        ),

        "final_answer": "",
    }


    # ========================================================
    # EXECUTE LANGGRAPH
    # ========================================================

    try:

        with logfire.span(
            "🧠 LangGraph Execution"
        ):

            final_output = (
                rag_agent.invoke(
                    initial_state,
                    config=config,
                )
            )


        # ====================================================
        # RESPONSE
        # ====================================================

        answer = final_output.get(
            "final_answer",
            "",
        )

        plan = final_output.get(
            "plan",
            [],
        )

        status = final_output.get(
            "status",
            "Completed.",
        )

        sources = final_output.get(
            "documents",
            [],
        )


        # ====================================================
        # EMPTY RESPONSE SAFETY
        # ====================================================

        if not answer:

            answer = (
                "I couldn't generate a response "
                "for that request."
            )


        logfire.info(
            "✅ LangGraph execution completed | "
            f"thread={thread_id}"
        )


        # ====================================================
        # RETURN
        # ====================================================

        return {

            "question": q,

            "answer": answer,

            "thought_process": plan,

            "status": status,

            "sources": sources,

            "thread_id": thread_id,
        }


    # ========================================================
    # BACKEND ERROR
    # ========================================================

    except Exception as exc:

        logfire.error(
            "❌ Backend execution failed: "
            f"{exc}"
        )

        return {

            "question": q,

            "answer": (
                "I apologize, but I encountered "
                "an internal error while processing "
                "your request. Please try again later."
            ),

            "thought_process": [
                "Execution error.",
                "Retrieval: Skipped",
            ],

            "status": "error",

            "sources": [],

            "thread_id": thread_id,
        }