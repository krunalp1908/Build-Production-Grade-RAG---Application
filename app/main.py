# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import logfire
import os
from dotenv import load_dotenv
import uuid

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# Now safe to import app modules - logfire is already active
from fastapi import FastAPI, Response
from app.agents.graph import rag_agent
from app.guardrails import initialize_rails, guard

from pydantic import BaseModel
from typing import Optional


# Initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API")


@app.on_event("startup")
def startup_event():
    initialize_rails()

class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = None


def save_guardrail_turn(config, user_message: str, assistant_message: str) -> None:
    """Store direct guardrail responses in the same conversation thread."""
    try:
        rag_agent.update_state(
            config,
            {"messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]},
        )
    except Exception as exc:
        logfire.warning(f"Could not save guardrail turn to memory: {exc}")


@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}


@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = (request.q or "").strip()
    thread_id = request.thread_id or str(uuid.uuid4())

    if not q:
        return {"question": q, "answer": "Please enter a question.",
                "thought_process": ["Guardrail: Empty request blocked"],
                "status": "No query provided.", "sources": [], "thread_id": thread_id}

    # Configuration for Memory (Thread ID)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        snapshot = rag_agent.get_state(config)
        prior_messages = snapshot.values.get("messages", []) if snapshot else []
    except Exception as exc:
        logfire.warning(f"Could not read session memory: {exc}")
        prior_messages = []

    try:
        # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
        rail_fired, rail_response, classification = guard(q, prior_messages)
        if rail_fired:
            save_guardrail_turn(config, q, rail_response)
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": [f"Guardrail: {classification}", "Retrieval: Skipped"],
                "status": "Handled by guardrails.",
                "sources": [], "thread_id": thread_id
            }

        initial_state = {
            "messages": [{"role": "user", "content": q}],
            "current_query": q,
            "intent": classification,
            "documents": [],
            "plan": [f"Guardrail: {classification}"],
            "status": "Initializing Graph..."
        }

        # Gate 2: LangGraph RAG pipeline
        # Run the graph synchronously to preserve Logfire context variables
        final_output = rag_agent.invoke(initial_state, config=config)

        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", []),
            "thread_id": thread_id
        }
    except Exception as e:
        logfire.error(f"❌ Backend Execution Failed: {e}")
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": []
        }
