import re

import logfire

from app.agents.state import AgentState
from app.gateway import get_langchain_llm


# ============================================================
# Planner LLM
# ============================================================

llm = get_langchain_llm(
    feature="planner"
)


# ============================================================
# Helper: extract message content
# ============================================================

def _message_content(message) -> str:

    if isinstance(message, dict):
        return str(
            message.get("content", "")
        )

    return str(
        getattr(message, "content", "")
    )


# ============================================================
# Planner Node
# ============================================================

def planner_node(state: AgentState):
    """
    Determines whether the user query requires RAG retrieval
    and resolves contextual follow-up questions using the
    conversation history.

    Important caching optimization:

    Exact duplicate questions are returned unchanged instead
    of being re-written by the LLM.

    This keeps repeated RAG requests deterministic and gives
    Portkey's cache a stable request to match.
    """

    messages = state.get(
        "messages",
        []
    )

    if state.get("intent") == "MEMORY":
        return {
            "current_query": "MEMORY",
            "intent": "MEMORY",
            "status": "Using conversation memory.",
            "plan": [
                "Guardrail: Memory request allowed",
                "Retrieval: Skipped",
            ],
        }

    if not messages:

        return {
            "current_query": "",
            "status": "No user message provided.",
            "plan": [
                "Intent: Empty request"
            ],
        }


    # ========================================================
    # Latest user message
    # ========================================================

    latest_message = messages[-1]

    user_message = _message_content(
        latest_message
    ).strip()

    # Normal in-scope questions already have all the information needed for
    # retrieval.  Keeping them unchanged gives Portkey's simple cache an
    # identical request on repeated questions.  Only ambiguous follow-ups
    # need an LLM rewrite using conversation history.
    contextual_reference = re.search(
        r"\b(it|this|that|they|them|previous|again)\b",
        user_message,
        re.IGNORECASE,
    )

    if not contextual_reference:
        return {
            "current_query": user_message,
            "intent": "RAG",
            "status": f"Searching for: {user_message}",
            "plan": [
                "Guardrail: RAG allowed",
                f"Search Term: {user_message}",
            ],
        }


    # ========================================================
    # Detect exact duplicate question
    # ========================================================
    #
    # If the user asks exactly the same question again,
    # don't ask the planner LLM to rewrite it.
    #
    # This is important for gateway caching.
    # ========================================================

    previous_user_messages = []

    for msg in messages[:-1]:

        role = (
            msg.get("role")
            if isinstance(msg, dict)
            else getattr(msg, "type", "")
        )

        if role in (
            "user",
            "human",
        ):

            previous_user_messages.append(
                _message_content(msg).strip()
            )


    if user_message in previous_user_messages:

        logfire.info(
            "♻️ Exact duplicate query detected. "
            "Keeping query unchanged for cache stability."
        )

        return {
            "current_query": user_message,

            "intent": "RAG",

            "status": (
                "Repeated query detected."
            ),

            "plan": [
                "Intent: Repeated RAG query",
                f"Search Term: {user_message}",
            ],
        }


    # ========================================================
    # Conversation history
    # ========================================================

    history_str = ""

    for msg in messages[:-1]:

        role = (
            msg.get("role")
            if isinstance(msg, dict)
            else getattr(msg, "type", "")
        )

        content = _message_content(msg)

        if role in (
            "user",
            "human",
        ):

            speaker = "User"

        elif role in (
            "assistant",
            "ai",
        ):

            speaker = "Assistant"

        else:

            speaker = str(role)


        history_str += (
            f"{speaker}: {content}\n"
        )


    # ========================================================
    # Planner prompt
    # ========================================================

    prompt = f"""
You are the query planner for a technical Enterprise RAG
assistant.

Your job is to determine whether the latest user message
should be answered using the enterprise technical knowledge
base.

The supported knowledge domains are:

- Kubernetes
- Intel
- Networking

CONVERSATION HISTORY:
{history_str if history_str else "(No previous conversation.)"}

LATEST USER MESSAGE:
{user_message}

RULES:

1. If the latest message is a greeting or farewell, output:

CONVERSATIONAL

Examples:
- hi
- hello
- hey
- good morning
- thanks, bye
- goodbye

2. If the latest message is a technical question about
   Kubernetes, Intel, or Networking, output a standalone
   search query.

3. If the latest message is a contextual follow-up, use the
   conversation history to resolve the missing context.

Example:

User:
What is a Kubernetes Deployment?

Assistant:
A Deployment manages Pods...

User:
Why is it useful?

Output:
Why is a Kubernetes Deployment useful?

4. Do not answer the question.

5. Do not explain your decision.

6. Output ONLY:

CONVERSATIONAL

OR

a standalone technical search query.
"""


    # ========================================================
    # LLM planning
    # ========================================================

    with logfire.span(
        "🧠 Planner Decision"
    ):

        decision = llm.invoke(
            prompt
        )

        decision = (
            decision.content
            if hasattr(
                decision,
                "content",
            )
            else str(decision)
        )

        decision = str(
            decision
        ).strip()


        logfire.info(
            f"Intent identified: {decision}"
        )


    # ========================================================
    # Conversational
    # ========================================================

    if decision.upper() == "CONVERSATIONAL":

        return {
            "current_query": "CONVERSATIONAL",

            "intent": "CONVERSATIONAL",

            "status": (
                "Handling conversational message."
            ),

            "plan": [
                "Intent: Conversational",
                "Retrieval: Skipped",
            ],
        }


    # ========================================================
    # Technical RAG
    # ========================================================

    return {
        "current_query": decision,

        "intent": "RAG",

        "status": (
            f"Technical research needed. "
            f"Searching for: {decision}"
        ),

        "plan": [
            "Intent: Technical RAG",
            f"Search Term: {decision}",
        ],
    }
