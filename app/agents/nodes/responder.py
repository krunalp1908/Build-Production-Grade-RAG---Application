import logfire
from app.agents.state import AgentState
from app.gateway import portkey_client, extract_cache_status


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Portkey client (not LangChain) so we can read the
    x-portkey-cache-status response header and surface Cache: Hit in the UI.
    """
    query = state["current_query"]

    def message_content(message):
        return str(message.get("content", "") if isinstance(message, dict) else getattr(message, "content", ""))

    def message_role(message):
        value = message.get("role", "") if isinstance(message, dict) else getattr(message, "type", "")
        return "User" if value in ("user", "human") else "Assistant"

    history_str = ""
    for msg in state["messages"][:-1]:
        history_str += f"{message_role(msg)}: {message_content(msg)}\n"

    user_msg = message_content(state["messages"][-1]) if state["messages"] else ""

    if state.get("intent") == "MEMORY":
        logfire.info("Generating response from session memory.")
        prompt = f"""
You are a friendly Enterprise IT Assistant.
Answer only from the session history. If the requested information is not
present, say that it is not available in this session.

SESSION HISTORY:
{history_str or "(No previous conversation.)"}

LATEST USER MESSAGE:
{user_msg}
"""
    elif query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        # The planner resolves ambiguous references. Excluding mutable session
        # history here keeps repeated RAG prompts identical for Portkey's
        # simple cache.
        prompt = f"""
You are a Senior Technical Architect and Enterprise RAG Assistant.
Answer using only the technical context below. Do not invent facts. If the
context is insufficient, say so clearly. Do not mention internal systems.

TECHNICAL CONTEXT:
{full_context}

USER QUESTION:
{query}
"""

    with logfire.span("✍️ LLM Synthesis"):
        try:
            response = portkey_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = response.choices[0].message.content
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                logfire.info("⚡ Gateway Cache Hit — response served from Portkey cache.")
                plan_update = state["plan"] + ["Cache: Hit ⚡"]
                status = "Cache hit — instant response."
            else:
                logfire.info("✅ Response synthesised via LLM.")
                plan_update = state["plan"]
                status = "Response generated."

            return {
                "final_answer": content,
                "status": status,
                "plan": plan_update,
                "messages": [{"role": "assistant", "content": content}]
            }

        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e
