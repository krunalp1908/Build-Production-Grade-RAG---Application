import logfire

from langchain_openai import ChatOpenAI
from portkey_ai import (
    createHeaders,
    PORTKEY_GATEWAY_URL,
)

from app.config import settings


# ============================================================
# Configuration validation
# ============================================================

def _validate_gateway_settings() -> None:

    missing = []

    if not settings.PORTKEY_API_KEY:
        missing.append("PORTKEY_API_KEY")

    if not settings.PORTKEY_CONFIG_ID:
        missing.append("PORTKEY_CONFIG_ID")

    if missing:
        raise RuntimeError(
            "Missing LLM Gateway configuration: "
            + ", ".join(missing)
        )


_validate_gateway_settings()


# ============================================================
# Application metadata
# ============================================================

GATEWAY_ENVIRONMENT = "resume-demo"

APPLICATION_NAME = "enterprise-agentic-rag"


# ============================================================
# LangChain → Portkey
# ============================================================

def get_langchain_llm(
    feature: str = "rag",
) -> ChatOpenAI:

    if not feature:
        feature = "rag"


    # --------------------------------------------------------
    # Portkey headers
    # --------------------------------------------------------

    headers = createHeaders(
        api_key=settings.PORTKEY_API_KEY,
        config=settings.PORTKEY_CONFIG_ID,
    )


    # --------------------------------------------------------
    # LangChain client
    # --------------------------------------------------------

    llm = ChatOpenAI(
    api_key=settings.PORTKEY_API_KEY,
    base_url=PORTKEY_GATEWAY_URL,
    model="gpt-4o-mini",
    temperature=0,
    default_headers=headers,
    )


    logfire.info(
        "🚪 Portkey Gateway initialized | "
        f"feature={feature} | "
        f"config={settings.PORTKEY_CONFIG_ID}"
    )


    return llm