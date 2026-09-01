import logfire

from app.agents.state import AgentState

from app.services.retrieval.qdrant_service import (
    search_enterprise_knowledge,
)

from app.services.retrieval.ranking_service import (
    rerank_documents,
)


# ============================================================
# RETRIEVER NODE
# ============================================================

def retrieve_node(
    state: AgentState,
):

    query = state.get(
        "current_query",
        "",
    )

    # ========================================================
    # Safety check
    # ========================================================

    if not query:

        return {
            "documents": [],
            "status": "No retrieval query available.",
            "plan": state.get(
                "plan",
                [],
            ) + [
                "Retrieval: Skipped"
            ],
        }

    with logfire.span(
        "🔍 Knowledge Retrieval"
    ):

        logfire.info(
            f"Searching Qdrant for: {query}"
        )

        # ----------------------------------------------------
        # Vector search
        # ----------------------------------------------------

        raw_results = (
            search_enterprise_knowledge(
                query,
                limit=15,
            )
        )

        logfire.info(
            f"Retrieved {len(raw_results)} "
            "candidates from Vector DB"
        )

        # ----------------------------------------------------
        # Extract document content
        # ----------------------------------------------------

        doc_contents = [
            doc["content"]
            for doc in raw_results
            if isinstance(doc, dict)
            and "content" in doc
        ]

        # ----------------------------------------------------
        # Semantic reranking
        # ----------------------------------------------------

        with logfire.span(
            "⚖️ Semantic Reranking"
        ):

            if doc_contents:

                reranked_contents = (
                    rerank_documents(
                        query,
                        doc_contents,
                        top_n=5,
                    )
                )

            else:

                reranked_contents = []

            logfire.info(
                "Reranking complete. "
                f"Kept {len(reranked_contents)} chunks."
            )

        # ----------------------------------------------------
        # Format context
        # ----------------------------------------------------

        formatted_docs = [
            f"CONTENT: {doc}"
            for doc in reranked_contents
        ]

    return {
        "documents": formatted_docs,

        "status": (
            f"Found {len(formatted_docs)} "
            "technical context chunks."
        ),

        "plan": state.get(
            "plan",
            [],
        ) + [
            "Context Retrieved"
        ],
    }