# ============================================================
# Load environment first
# ============================================================

import os

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Logfire
# ============================================================

import logfire

logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN")
)


# ============================================================
# FastAPI
# ============================================================

from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import Optional


# ============================================================
# Application
# ============================================================

from app.agents.graph import rag_agent

from app.guardrails import (
    initialize_rails,
    guard,
)


# ============================================================
# FastAPI application
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
        "message": (
            "Enterprise LangGraph RAG API is live."
        )
    }


# ============================================================
# Graph visualization
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

        return {
            "error": (
                f"Could not generate graph image: {exc}"
            )
        }


# ============================================================
# Query endpoint
# ============================================================

@app.post("/query")
def query(
    request: QueryRequest
):

    q = (
        request.q or ""
    ).strip()


    thread_id = (
        request.thread_id
        or "default_user"
    )


    # ========================================================
    # Empty input
    # ========================================================

    if not q:

        return {
            "question": q,

            "answer": (
                "Please enter a question."
            ),

            "thought_process": [
                "Empty request."
            ],

            "status": "No query provided.",

            "sources": [],
        }


    # ========================================================
    # LangGraph session configuration
    # ========================================================

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }


    # ========================================================
    # READ EXISTING SESSION MEMORY
    # ========================================================
    #
    # This happens BEFORE the guardrail.
    #
    # The guardrail needs previous messages so it can
    # understand contextual follow-up questions.
    #
    # Example:
    #
    # Previous:
    #   What is a Kubernetes CronJob?
    #
    # Current:
    #   How often does it run?
    #
    # The guardrail can determine that "it" refers to
    # Kubernetes CronJob.
    # ========================================================

    try:

        state_snapshot = rag_agent.get_state(
            config
        )

        existing_state = (
            state_snapshot.values
            if state_snapshot
            else {}
        )

    except Exception as exc:

        logfire.error(
            f"❌ Could not retrieve session memory: {exc}"
        )

        existing_state = {}


    # ========================================================
    # Previous messages
    # ========================================================

    prior_messages = (
        existing_state.get(
            "messages",
            []
        )
        if isinstance(
            existing_state,
            dict,
        )
        else []
    )


    # ========================================================
    # GUARDRAIL
    # ========================================================
    #
    # The guard receives:
    #
    #   current question
    #           +
    #   previous session messages
    #
    # It can therefore distinguish:
    #
    #   "What is Kubernetes?"
    #
    # from:
    #
    #   "How does it work?"
    #
    # when the latter follows a Kubernetes discussion.
    # ========================================================

    try:

        rail_fired, rail_response = guard(
            message=q,
            prior_messages=prior_messages,
        )

    except Exception as exc:

        logfire.error(
            f"❌ Guardrail execution failed: {exc}"
        )

        # ----------------------------------------------------
        # FAIL CLOSED
        # ----------------------------------------------------

        return {
            "question": q,

            "answer": (
                "I’m sorry, but I couldn’t safely "
                "process that request right now."
            ),

            "thought_process": [
                "Intent: Guardrail failure",
                "Retrieval: Skipped",
            ],

            "status": (
                "Blocked by safety guardrail."
            ),

            "sources": [],
        }


    # ========================================================
    # Guardrail handled the request
    # ========================================================

    if rail_fired:

        logfire.info(
            "🛡️ Request handled by guardrail | "
            f"thread={thread_id}"
        )

        return {
            "question": q,

            "answer": rail_response,

            "thought_process": [
                "Intent: Guardrails",
                "Retrieval: Skipped",
            ],

            "status": (
                "Handled by guardrails."
            ),

            "sources": [],
        }


    # ========================================================
    # NEW GRAPH INPUT
    # ========================================================
    #
    # IMPORTANT:
    #
    # Do NOT manually include prior_messages here.
    #
    # Your AgentState has:
    #
    # messages: Annotated[List[dict], operator.add]
    #
    # Therefore LangGraph's checkpoint/reducer is responsible
    # for accumulating the conversation.
    # ========================================================

    initial_state = {

        "messages": [
            {
                "role": "user",
                "content": q,
            }
        ],

        "current_query": q,

        "documents": [],

        "plan": [
            "Start"
        ],

        "status": (
            "Initializing Graph..."
        ),

        "final_answer": "",
    }


    # ========================================================
    # RUN LANGGRAPH
    # ========================================================

    try:

        final_output = rag_agent.invoke(
            initial_state,
            config=config,
        )


        # ====================================================
        # Response
        # ====================================================

        return {

            "question": q,

            "answer": final_output.get(
                "final_answer",
                "",
            ),

            "thought_process": final_output.get(
                "plan",
                [],
            ),

            "status": final_output.get(
                "status",
                "",
            ),

            "sources": final_output.get(
                "documents",
                [],
            ),
        }


    except Exception as exc:

        logfire.error(
            f"❌ Backend execution failed: {exc}"
        )

        return {

            "question": q,

            "answer": (
                "I apologize, but I encountered "
                "an internal error while processing "
                "your request. Please try again later."
            ),

            "thought_process": [
                "Error encountered during execution."
            ],

            "status": "error",

            "sources": [],
        }