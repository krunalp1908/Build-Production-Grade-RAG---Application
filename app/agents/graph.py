from langgraph.graph import (
    StateGraph,
    END,
)

from langgraph.checkpoint.memory import MemorySaver

from app.agents.state import AgentState

from app.agents.nodes.planner import (
    planner_node,
)

from app.agents.nodes.retriever import (
    retrieve_node,
)

from app.agents.nodes.responder import (
    generate_node,
)


# ============================================================
# GRAPH
# ============================================================

workflow = StateGraph(
    AgentState
)


# ============================================================
# NODES
# ============================================================

workflow.add_node(
    "planner",
    planner_node,
)

workflow.add_node(
    "retriever",
    retrieve_node,
)

workflow.add_node(
    "responder",
    generate_node,
)


# ============================================================
# ROUTER
# ============================================================

def route_after_planner(
    state: AgentState,
):

    intent = state.get(
        "intent",
        "OUT_OF_SCOPE",
    )

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    if intent == "RAG":

        return "retriever"

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    if intent == "MEMORY":

        return "responder"

    # --------------------------------------------------------
    # Conversational
    # --------------------------------------------------------

    if intent == "CONVERSATIONAL":

        return "responder"

    # --------------------------------------------------------
    # Everything else is blocked.
    #
    # It goes to responder only so responder can return the
    # deterministic blocked message. It will NOT call the LLM.
    # --------------------------------------------------------

    return "responder"


# ============================================================
# ENTRY POINT
# ============================================================

workflow.set_entry_point(
    "planner"
)


# ============================================================
# PLANNER ROUTING
# ============================================================

workflow.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "retriever": "retriever",
        "responder": "responder",
    },
)


# ============================================================
# RAG → RESPONDER
# ============================================================

workflow.add_edge(
    "retriever",
    "responder",
)


# ============================================================
# RESPONDER → END
# ============================================================

workflow.add_edge(
    "responder",
    END,
)


# ============================================================
# SESSION MEMORY
# ============================================================

checkpointer = MemorySaver()


# ============================================================
# COMPILE
# ============================================================

rag_agent = workflow.compile(
    checkpointer=checkpointer,
)