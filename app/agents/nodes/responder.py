import logfire

from app.agents.state import AgentState
from app.gateway import get_langchain_llm


# ============================================================
# Responder LLM
# ============================================================

llm = get_langchain_llm(
    feature="responder"
)


# ============================================================
# Responder Node
# ============================================================

def generate_node(state: AgentState):
    """
    Generates the final answer from retrieved enterprise
    documentation.

    Conversation history is intentionally NOT included in the
    final RAG prompt.

    The planner already uses conversation memory to resolve
    contextual questions.

    Keeping the final prompt deterministic improves the
    effectiveness of Portkey's cache.
    """

    query = state["current_query"]
    intent = state.get("intent", "RAG")

    if intent == "MEMORY":
        def content(message):
            return str(
                message.get("content", "")
                if isinstance(message, dict)
                else getattr(message, "content", "")
            )

        def role(message):
            value = (
                message.get("role", "")
                if isinstance(message, dict)
                else getattr(message, "type", "")
            )
            return "User" if value in ("user", "human") else "Assistant"

        history = "\n".join(
            f"{role(message)}: {content(message)}"
            for message in state.get("messages", [])[:-1]
        )

        latest_user_message = content(state.get("messages", [])[-1])

        prompt = f"""
You are a friendly Enterprise IT Assistant.
Answer only from this session history. If the information is absent, say so.

SESSION HISTORY:
{history or "(No previous conversation.)"}

LATEST USER MESSAGE:
{latest_user_message}
"""

    # ========================================================
    # Conversational response
    # ========================================================
    #
    # This is retained for compatibility with the existing
    # graph.
    #
    # Ideally your guardrail handles greetings/farewells
    # before the graph.
    # ========================================================

    elif query == "CONVERSATIONAL":

        prompt = """
You are a friendly Enterprise AI Assistant.

Respond naturally to the user's conversational message.

Be brief, warm, and human-like.

Do not discuss internal implementation details.
"""


    else:

        # ====================================================
        # Technical RAG response
        # ====================================================

        logfire.info(
            "Generating technical RAG response."
        )


        # ====================================================
        # Build deterministic context
        # ====================================================

        max_context_chars = 25000

        full_context = ""


        for doc in state.get(
            "documents",
            [],
        ):

            doc_text = str(
                doc
            )


            if (
                len(full_context)
                + len(doc_text)
                + 2
                <= max_context_chars
            ):

                full_context += (
                    doc_text
                    + "\n\n"
                )

            else:

                logfire.warning(
                    "Context truncated to fit "
                    "LLM limits."
                )

                break


        # ====================================================
        # RAG prompt
        # ====================================================
        #
        # IMPORTANT:
        #
        # No conversation history.
        #
        # The planner has already resolved it.
        #
        # This makes repeated requests much more cacheable.
        # ====================================================

        prompt = f"""
You are a friendly Senior Technical Architect working as an
Enterprise RAG Assistant.

Answer the user's question using ONLY the technical
documentation provided below.

TECHNICAL CONTEXT:
{full_context}

USER QUESTION:
{query}

RULES:

1. Use the technical context as the primary source of truth.

2. Do not invent technical facts that are not supported by
   the provided documentation.

3. If the documentation does not contain enough information,
   clearly say that the available documentation does not
   contain enough information.

4. Answer naturally, like a knowledgeable human technical
   colleague.

5. Do not mention guardrails, LangGraph, Portkey, prompts,
   retrieval pipelines, or internal implementation details.

6. Do not unnecessarily repeat the question.

7. Use examples when they make the explanation clearer.

8. Keep the answer focused on the user's question.
"""


    # ========================================================
    # LLM generation
    # ========================================================

    with logfire.span(
        "✍️ LLM Synthesis"
    ):

        try:

            response = llm.invoke(
                prompt
            )


            # =================================================
            # Extract content
            # =================================================

            content = (
                response.content
                if hasattr(
                    response,
                    "content",
                )
                else str(response)
            )


            content = str(
                content
            ).strip()


            # =================================================
            # Update execution plan
            # =================================================

            previous_plan = state.get(
                "plan",
                []
            )


            plan_update = list(
                previous_plan
            )


            plan_update.append(
                "Response generated via Portkey Gateway"
            )


            logfire.info(
                "✅ Response synthesised via "
                "Portkey Gateway."
            )


            # =================================================
            # Return
            # =================================================

            return {

                "final_answer": content,

                "status": (
                    "Response generated."
                ),

                "plan": plan_update,

                "messages": [
                    {
                        "role": "assistant",
                        "content": content,
                    }
                ],
            }


        except Exception as exc:

            logfire.error(
                f"❌ LLM Generation failed: {exc}"
            )

            raise
