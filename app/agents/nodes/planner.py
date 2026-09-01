from app.agents.state import AgentState
from app.config import settings
from langchain_groq import ChatGroq
import logfire


# ============================================================
# Planner LLM
# ============================================================

llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model=settings.GROQ_MODEL,
    temperature=0,
)


def planner_node(state: AgentState):
    """
    Planner converts an allowed user request into a focused
    retrieval query.

    Guardrails are responsible for deciding whether the request
    belongs to the RAG domain. The planner only operates after
    the request has passed that guardrail.
    """

    messages = state.get("messages", [])

    # --------------------------------------------------------
    # Extract previous conversation
    # --------------------------------------------------------

    history_messages = messages[:-1]

    history = ""

    for msg in history_messages:

        # Support both dictionary-style messages and
        # LangChain-style message objects.
        if isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", "")

        if role in ("user", "human"):
            speaker = "User"
        elif role in ("assistant", "ai"):
            speaker = "Assistant"
        else:
            speaker = role

        history += f"{speaker}: {content}\n"


    # --------------------------------------------------------
    # Latest user message
    # --------------------------------------------------------

    if messages:

        latest_message = messages[-1]

        if isinstance(latest_message, dict):
            user_message = latest_message.get(
                "content",
                ""
            )
        else:
            user_message = getattr(
                latest_message,
                "content",
                ""
            )

    else:
        user_message = ""


    # --------------------------------------------------------
    # Planner prompt
    # --------------------------------------------------------

    prompt = f"""
You are the retrieval planner for an enterprise RAG chatbot.

The request has already passed the application's
RAG guardrail.

Your ONLY job is to create a concise search query that
can be sent to the retrieval system.

The knowledge base covers:

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
- CPUs
- FPGAs
- NICs
- SR-IOV
- Enterprise networking
- SDN
- VLANs
- BGP
- Routing
- Related infrastructure operations

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

Create the best concise retrieval query for the latest
user request.

Use the conversation history only to resolve references
such as:

- "it"
- "that"
- "this"
- "the previous one"
- "how does that work?"
- "what happens next?"

For example:

Conversation:
User: What is a Kubernetes CronJob?
Assistant: A CronJob creates Jobs on a schedule.

Latest message:
How often does it run?

Good search query:
Kubernetes CronJob scheduling frequency

Another example:

Conversation:
User: What is SR-IOV?
Assistant: [explanation]

Latest message:
What are its benefits?

Good search query:
SR-IOV benefits

Do NOT answer the question.

Do NOT write explanations.

Do NOT include greetings.

Return ONLY the search query.
"""


    # --------------------------------------------------------
    # LLM call
    # --------------------------------------------------------

    with logfire.span("🧠 Planner Decision"):

        try:

            decision = (
                llm
                .invoke(prompt)
                .content
                .strip()
            )

            logfire.info(
                f"🔎 Retrieval query generated: {decision}"
            )

        except Exception as exc:

            logfire.error(
                f"❌ Planner failed: {exc}"
            )

            # Simple fallback for a testing/demo project.
            decision = user_message


    # --------------------------------------------------------
    # Return planner state
    # --------------------------------------------------------

    return {
        "current_query": decision,
        "status": (
            f"Technical research needed. "
            f"Searching for: {decision}"
        ),
        "plan": [
            "Intent: RAG",
            f"Search Term: {decision}",
        ],
    }