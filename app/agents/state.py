from typing import TypedDict, List, Annotated
import operator


class AgentState(TypedDict):
    # Messages are accumulated across graph executions
    # for the same LangGraph thread_id.
    messages: Annotated[List[dict], operator.add]

    current_query: str

    documents: List[str]

    plan: List[str]

    status: str

    final_answer: str