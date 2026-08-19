import logfire
import re
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


_rails: LLMRails | None = None

OFF_TOPIC_RESPONSE = (
    "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, "
    "and networking. I can't help with that, but ask me a technical question!"
)
OFF_TOPIC_TERMS = re.compile(
    r"\b(recipe|cooking|cook|coffee|food|dinner|breakfast|lunch|restaurant|"
    r"movie|poem|joke|weather|math homework|world history)\b",
    re.IGNORECASE,
)


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses the configurable safeguard model for intent classification at the gate;
    the heavier RAG model is reserved for the answer-generation pipeline.
    """
    global _rails

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_GUARD_MODEL,
        temperature=0
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info(f"🛡️ NeMo Guardrails initialised ({settings.GROQ_GUARD_MODEL}).")
    
    


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if OFF_TOPIC_TERMS.search(message):
        logfire.info(f"🛡️ Deterministic off-topic rail fired | query='{message[:80]}'")
        return True, OFF_TOPIC_RESPONSE

    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        result = _rails.generate(messages=[{"role": "user", "content": message}])

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None
