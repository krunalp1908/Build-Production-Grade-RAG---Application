import logfire
from langchain_openai import ChatOpenAI
from nemoguardrails import LLMRails, RailsConfig

from app.config import settings
from app.guardrails.colang_rules import (
    CAPABILITIES_RESPONSE,
    CAPABILITY_KEYWORDS,
    COLANG_CONTENT,
    FAREWELL_KEYWORDS,
    FAREWELL_RESPONSE,
    GREETING_KEYWORDS,
    GREETING_RESPONSE,
    KNOWN_RAIL_RESPONSES,
    MEMORY_NAME_PATTERNS,
    MEMORY_QUESTION_PATTERNS,
    OFF_TOPIC_PATTERNS,
    OFF_TOPIC_RESPONSE,
    RAIL_INDICATORS,
    TECHNICAL_KEYWORDS,
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


def guard(message: str, history: list[str] | None = None) -> tuple[bool, str | None]:
    """
    Simple guardrail policy:
      - allow greetings / capabilities / farewells
      - allow technical knowledge-base questions
      - block known off-topic prompts
      - use session history so the last question is remembered
    """
    text = (message or "").strip()
    if not text:
        logfire.warning("🛡️ Empty guardrail input — blocking request.")
        return True, OFF_TOPIC_RESPONSE

    history_text = " ".join((h or "") for h in (history or []) if h)
    combined = f"{history_text} {text}".lower()
    normalized = text.lower()

    if any(keyword in normalized for keyword in GREETING_KEYWORDS):
        return True, GREETING_RESPONSE

    if any(keyword in normalized for keyword in CAPABILITY_KEYWORDS):
        return True, CAPABILITIES_RESPONSE

    if any(keyword in normalized for keyword in FAREWELL_KEYWORDS):
        return True, FAREWELL_RESPONSE

    if any(memory_pattern in normalized for memory_pattern in MEMORY_NAME_PATTERNS):
        logfire.info("✅ Personal memory statement detected; allowing conversational memory.")
        return False, None

    if any(pattern in normalized for pattern in OFF_TOPIC_PATTERNS):
        logfire.info("🛡️ Off-topic question detected; firing guardrail.")
        return True, OFF_TOPIC_RESPONSE

    if any(question in combined for question in MEMORY_QUESTION_PATTERNS):
        if any(memory_pattern in history_text for memory_pattern in MEMORY_NAME_PATTERNS):
            logfire.info("✅ Memory-based conversational question detected; allowing conversation.")
            return False, None

    if any(keyword in combined for keyword in TECHNICAL_KEYWORDS):
        logfire.info("✅ Technical knowledge-base question detected; allowing RAG.")
        return False, None

    if _rails is not None:
        with logfire.span("🛡️ Guardrails Check"):
            try:
                result = _rails.generate(messages=[{"role": "user", "content": text}])
            except Exception:
                logfire.exception("🛡️ Guardrails evaluation failed — blocking request.")
                return True, OFF_TOPIC_RESPONSE

            content = result.get("content", "") if isinstance(result, dict) else str(result)
            content = (content or "").strip()
            if TECHNICAL_QUERY_ALLOWED in content:
                logfire.info("✅ Technical query accepted by guardrails.")
                return False, None

            fired = any(indicator.lower() in content.lower() for indicator in RAIL_INDICATORS)
            if fired:
                logfire.info(f"🛡️ Guardrails fired | query='{text[:80]}'")
                return True, next(
                    (
                        known_response
                        for known_response in KNOWN_RAIL_RESPONSES
                        if known_response.lower() in content.lower()
                    ),
                    OFF_TOPIC_RESPONSE,
                )

    logfire.warning("🛡️ Unclassified request; blocking with technical refusal.")
    return True, OFF_TOPIC_RESPONSE
