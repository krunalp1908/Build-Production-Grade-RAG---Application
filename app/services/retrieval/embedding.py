import threading

import logfire
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 50

_model: SentenceTransformer | None = None
_lock = threading.Lock()


# ------------------------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------------------------

def _init() -> None:
    """
    Lazily load the embedding model once per process.
    """
    global _model

    if _model is not None:
        return

    with _lock:
        if _model is not None:
            return

        logfire.info(f"Loading embedding model: {MODEL_NAME}")

        _model = SentenceTransformer(MODEL_NAME)

        logfire.info(
            f"Embedding model loaded successfully: {MODEL_NAME} "
            f"(dimension={_model.get_sentence_embedding_dimension()})"
        )


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def get_embedding_dim() -> int:
    """
    Return the embedding dimension of the active model.
    """
    _init()
    return _model.get_sentence_embedding_dimension()


def get_model_name() -> str:
    """
    Return the embedding model name.
    """
    return MODEL_NAME


def _encode(texts: list[str]) -> list[list[float]]:
    _init()

    embeddings = _model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.tolist()


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------

def embed_query(query: str) -> list[float]:
    vector = _encode([query])[0]

    logfire.info(
        "Embedding Debug",
        vector_type=str(type(vector)),
        first_element_type=str(type(vector[0])) if hasattr(vector, "__getitem__") else "N/A",
        length=len(vector) if hasattr(vector, "__len__") else 0,
    )

    return vector


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts in batches.
    """
    embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        with logfire.span(
            "Embed batch",
            model=MODEL_NAME,
            start=i,
            size=len(batch),
        ):
            embeddings.extend(_encode(batch))

    return embeddings