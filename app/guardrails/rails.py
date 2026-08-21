import logfire
from langchain_openai import ChatOpenAI
from nemoguardrails import LLMRails, RailsConfig

from app.config import settings
from app.guardrails.colang_rules import (
    COLANG_CONTENT,
    KNOWN_RAIL_RESPONSES,
    OFF_TOPIC_RESPONSE,
    RAIL_INDICATORS,
    TECHNICAL_QUERY_ALLOWED,
    YAML_CONTENT,
)

_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses OpenAI gpt-4o-mini for fast intent classification at the gate.
    """
    global _rails

    guard_llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY, model="gpt-4o-mini")

    config = RailsConfig.from_content(colang_content=COLANG_CONTENT, yaml_content=YAML_CONTENT)

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info("🛡️ NeMo Guardrails initialised (gpt-4o-mini).")


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if _rails is None:
        logfire.error("🛡️ Guardrails not initialised — blocking request.")
        return True, OFF_TOPIC_RESPONSE

    with logfire.span("🛡️ Guardrails Check"):
        try:
            result = _rails.generate(messages=[{"role": "user", "content": message}])
        except Exception:
            logfire.exception("🛡️ Guardrails evaluation failed — blocking request.")
            return True, OFF_TOPIC_RESPONSE

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text.
        content = result.get("content", "") if isinstance(result, dict) else str(result)
        content = content.strip()

        if TECHNICAL_QUERY_ALLOWED in content:
            logfire.info("✅ Technical query accepted by guardrails.")
            return False, None

        fired = any(indicator.lower() in content.lower() for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            response = next(
                (
                    known_response
                    for known_response in KNOWN_RAIL_RESPONSES
                    if known_response.lower() in content.lower()
                ),
                OFF_TOPIC_RESPONSE,
            )
            return True, response

        # Fail closed: an unclassified response must not reach the RAG pipeline.
        logfire.warning("🛡️ Guardrails returned an unclassified intent; blocking request.")
        return True, OFF_TOPIC_RESPONSE
