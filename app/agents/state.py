from typing import TypedDict, Annotated, List, Any
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """
    Shared state used by the LangGraph Agent.

    `add_messages` is important because LangGraph uses it to
    accumulate conversation messages across the same thread_id.

    This allows MemorySaver to maintain:

        User -> Assistant -> User -> Assistant -> ...

    instead of replacing the previous conversation.
    """

    # ========================================================
    # Conversation Memory
    # ========================================================

    messages: Annotated[
        List[Any],
        add_messages,
    ]

    # ========================================================
    # Planner / Routing
    # ========================================================

    current_query: str

    intent: str

    # ========================================================
    # RAG
    # ========================================================

    documents: List[str]

    # ========================================================
    # UI / Observability
    # ========================================================

    plan: List[str]

    status: str

    # ========================================================
    # Final Response
    # ========================================================

    final_answer: str