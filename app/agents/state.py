from typing import Any, Annotated, List, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # Using Annotated with operator.add ensures that messages
    # are appended to the history rather than replaced.
    messages: Annotated[List[Any], add_messages]
    current_query: str
    intent: str
    documents: List[str]
    plan: List[str]
    status: str
    final_answer: str
