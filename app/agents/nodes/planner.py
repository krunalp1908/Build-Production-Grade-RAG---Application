import logfire

from langchain_groq import ChatGroq

from app.agents.state import AgentState
from app.config import settings


# ============================================================
# PLANNER LLM
# ============================================================

llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model=settings.GROQ_MODEL,
    temperature=0,
)


# ============================================================
# MESSAGE HELPERS
# ============================================================

def _get_content(message) -> str:
    """
    Works with both:
        dict messages
    and:
        LangChain HumanMessage / AIMessage
    """

    if isinstance(
        message,
        dict,
    ):

        return str(
            message.get(
                "content",
                "",
            )
        )

    return str(
        getattr(
            message,
            "content",
            "",
        )
    )


def _get_role(message) -> str:

    if isinstance(
        message,
        dict,
    ):

        role = message.get(
            "role",
            "unknown",
        )

    else:

        role = getattr(
            message,
            "type",
            "unknown",
        )

    if role in (
        "human",
        "user",
    ):

        return "User"

    if role in (
        "ai",
        "assistant",
    ):

        return "Assistant"

    return str(role)


def _build_history(
    messages,
) -> str:

    history = []

    for message in messages[:-1]:

        role = _get_role(
            message
        )

        content = _get_content(
            message
        )

        history.append(
            f"{role}: {content}"
        )

    return "\n".join(
        history
    )


# ============================================================
# PLANNER
# ============================================================

def planner_node(
    state: AgentState,
):

    messages = state.get(
        "messages",
        [],
    )

    if not messages:

        return {
            "current_query": "OUT_OF_SCOPE",
            "intent": "OUT_OF_SCOPE",
            "status": "No message available.",
            "plan": [
                "Guardrail: Blocked",
                "Retrieval: Skipped",
                "LLM synthesis: Skipped",
            ],
        }

    # ========================================================
    # Latest user message
    # ========================================================

    latest_message = messages[-1]

    user_message = _get_content(
        latest_message
    )

    # ========================================================
    # Conversation history
    # ========================================================

    history = _build_history(
        messages
    )

    # ========================================================
    # Intent supplied by application guardrail
    # ========================================================

    guard_intent = state.get(
        "intent",
        "RAG",
    )

    # ========================================================
    # MEMORY
    # ========================================================

    if guard_intent == "MEMORY":

        logfire.info(
            "🧠 Planner: memory request"
        )

        return {
            "current_query": "MEMORY",
            "intent": "MEMORY",
            "status": (
                "Using conversation memory."
            ),
            "plan": [
                "Guardrail: Memory request allowed",
                "Retrieval: Skipped",
            ],
        }

    # ========================================================
    # RAG
    # ========================================================

    if guard_intent == "RAG":

        rewrite_prompt = f"""
You are a retrieval query planner for an enterprise
documentation-grounded RAG assistant.

The user's request has already passed the application's
strict security and relevance guardrail.

Your ONLY task is to rewrite the latest user request into
a concise retrieval query.

You MUST NOT answer the user.

You MUST NOT provide an explanation.

You MUST NOT use your own knowledge.

Use the conversation history ONLY to resolve references such as:

- it
- this
- that
- they
- them
- the previous one
- the previous configuration
- explain that again
- how does that work?

============================================================
KNOWLEDGE-BASE DOMAIN
============================================================

The knowledge base focuses on:

- Kubernetes
- Pods
- Deployments
- Services
- Jobs
- CronJobs
- Scheduling
- Autoscaling
- Workload management
- Kubernetes networking
- Kubernetes operators
- Intel hardware
- Intel CPUs
- FPGAs
- NICs
- SR-IOV
- Enterprise networking
- SDN
- VLANs
- BGP
- Routing
- Related enterprise infrastructure

============================================================
CONVERSATION HISTORY
============================================================

{history if history else "(No previous conversation.)"}

============================================================
LATEST USER MESSAGE
============================================================

{user_message}

============================================================
TASK
============================================================

Return ONLY a concise search query.

Example:

User:
"What is a Kubernetes CronJob?"

Return:

Kubernetes CronJob

Example:

Previous:
"What is SR-IOV?"

Current:
"What are its benefits?"

Return:

SR-IOV benefits
"""

        with logfire.span(
            "🧠 Retrieval Query Planner"
        ):

            try:

                search_query = (
                    llm
                    .invoke(
                        rewrite_prompt
                    )
                    .content
                    .strip()
                )

            except Exception as exc:

                logfire.error(
                    f"❌ Planner failed: {exc}"
                )

                # Safe fallback for the demo project.
                search_query = user_message

        if not search_query:

            search_query = user_message

        logfire.info(
            f"🔎 Retrieval query: {search_query}"
        )

        return {
            "current_query": search_query,
            "intent": "RAG",
            "status": (
                "Technical research needed. "
                f"Searching for: {search_query}"
            ),
            "plan": [
                "Guardrail: RAG allowed",
                f"Search Term: {search_query}",
            ],
        }

    # ========================================================
    # FAIL CLOSED
    # ========================================================

    return {
        "current_query": "OUT_OF_SCOPE",
        "intent": "OUT_OF_SCOPE",
        "status": (
            "Request did not pass the RAG guardrail."
        ),
        "plan": [
            "Guardrail: Request blocked",
            "Retrieval: Skipped",
            "LLM synthesis: Skipped",
        ],
    }