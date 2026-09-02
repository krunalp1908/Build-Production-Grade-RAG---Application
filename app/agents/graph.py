from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.agents.state import AgentState

from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.responder import generate_node


# ============================================================
# Create workflow
# ============================================================

workflow = StateGraph(
    AgentState
)


# ============================================================
# Nodes
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
# Graph flow
# ============================================================

def route_after_planner(state: AgentState):
    """Memory questions do not need retrieval; RAG questions do."""

    if state.get("intent") == "MEMORY":
        return "responder"

    return "retriever"


workflow.set_entry_point("planner")

workflow.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "retriever": "retriever",
        "responder": "responder",
    },
)


workflow.add_edge(
    "retriever",
    "responder",
)


workflow.add_edge(
    "responder",
    END,
)


# ============================================================
# Session Memory
# ============================================================

checkpointer = MemorySaver()


# ============================================================
# Compile graph
# ============================================================

rag_agent = workflow.compile(
    checkpointer=checkpointer
)
