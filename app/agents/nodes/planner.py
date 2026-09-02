from app.agents.state import AgentState
from app.gateway import get_langchain_llm
import logfire
import re

# Portkey-backed LLM: fallback + cache + retry — same .invoke() interface as ChatGroq
llm = get_langchain_llm(feature="planner")

def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation.
    """
    if state.get("intent") == "MEMORY":
        return {
            "current_query": "MEMORY",
            "intent": "MEMORY",
            "status": "Using conversation memory.",
            "plan": ["Guardrail: Memory request allowed", "Retrieval: Skipped"],
        }

    # Get the conversation history (excluding the latest message)
    history = ""
    for msg in state["messages"][:-1]:
        if isinstance(msg, dict):
            message_role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            message_role = getattr(msg, "type", "")
            content = getattr(msg, "content", "")
        role = "User" if message_role in ("user", "human") else "Assistant"
        history += f"{role}: {content}\n"

    latest = state["messages"][-1] if state["messages"] else {}
    user_message = latest.get("content", "") if isinstance(latest, dict) else getattr(latest, "content", "")

    if not re.search(r"\b(it|this|that|they|them|previous|again)\b", user_message, re.IGNORECASE):
        return {
            "current_query": user_message,
            "intent": "RAG",
            "status": f"Technical research needed. Searching for: {user_message}",
            "plan": ["Guardrail: RAG allowed", f"Search Term: {user_message}"],
        }

    prompt = f"""
    You are an intelligent Assistant Planner.
    Analyze the conversation history and the latest user message.

    CONVERSATION HISTORY:
    {history}

    LATEST MESSAGE:
    "{user_message}"

    Task:
    1. If the latest message is a greeting (hi, hello) or a question that can be answered using ONLY the conversation history above (e.g., "what is my name"), respond with 'CONVERSATIONAL'.
    2. If it is a technical question about Kubernetes, Intel, or Networking that requires fresh documentation, output a refined search query.

    Output ONLY 'CONVERSATIONAL' or the search query.
    """

    with logfire.span("🧠 Planner Decision"):
        decision = llm.invoke(prompt).content.strip()
        logfire.info(f"Intent identified: {decision}")

    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "intent": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }

    return {
        "current_query": decision,
        "intent": "RAG",
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"]
    }
