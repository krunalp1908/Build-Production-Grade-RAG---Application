from typing import Any, Annotated, List, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # Messages are accumulated across graph executions
    # for the same LangGraph thread_id.
    messages: Annotated[List[Any], add_messages]

    current_query: str

    intent: str

    documents: List[str]

    plan: List[str]

    status: str

    final_answer: str
