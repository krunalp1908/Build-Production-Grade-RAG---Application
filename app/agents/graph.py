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
#
# Guardrails happen BEFORE this graph.
#
# Therefore every request entering this graph has already
# passed the RAG relevance gate.
# ============================================================

workflow.set_entry_point(
    "planner"
)


workflow.add_edge(
    "planner",
    "retriever",
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