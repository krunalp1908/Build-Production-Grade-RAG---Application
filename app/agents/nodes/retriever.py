import logfire

from app.agents.state import AgentState

from app.services.retrieval.qdrant_service import (
    search_enterprise_knowledge,
)

from app.services.retrieval.ranking_service import (
    rerank_documents,
)


def retrieve_node(state: AgentState):
    """
    Retrieves and reranks technical documentation.

    The final document order is kept deterministic so that
    identical questions can produce identical responder
    prompts, allowing Portkey cache hits.
    """

    query = state["current_query"]


    # ========================================================
    # Retrieval
    # ========================================================

    with logfire.span(
        "🔍 Knowledge Retrieval"
    ):

        logfire.info(
            f"Searching Qdrant for: {query}"
        )


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


        if not raw_results:

            return {
                "documents": [],

                "status": (
                    "No technical context found."
                ),

                "plan": (
                    state["plan"]
                    + ["Context Retrieved: 0 chunks"]
                ),
            }


        # ====================================================
        # Extract document contents
        # ====================================================

        doc_contents = []

        for doc in raw_results:

            content = doc.get(
                "content",
                "",
            )

            if content:

                doc_contents.append(
                    str(content).strip()
                )


        # ====================================================
        # Semantic reranking
        # ====================================================

        with logfire.span(
            "⚖️ Semantic Reranking"
        ):

            reranked_contents = (
                rerank_documents(
                    query,
                    doc_contents,
                    top_n=5,
                )
            )


            logfire.info(
                "Reranking complete. "
                "Kept top 5 most relevant chunks."
            )


        # ====================================================
        # Clean + deterministic context
        # ========================================================

        cleaned_documents = []

        for doc in reranked_contents:

            text = str(
                doc
            ).strip()

            if text and text not in cleaned_documents:

                cleaned_documents.append(
                    text
                )


        # ====================================================
        # Format documents
        # ========================================================

        formatted_docs = []

        for index, doc in enumerate(
            cleaned_documents,
            start=1,
        ):

            formatted_docs.append(
                f"DOCUMENT {index}:\n{doc}"
            )


    # ========================================================
    # Return state update
    # ========================================================

    return {

        "documents": formatted_docs,

        "status": (
            f"Found {len(formatted_docs)} "
            "technical context chunks."
        ),

        "plan": (
            state["plan"]
            + [
                (
                    f"Context Retrieved: "
                    f"{len(formatted_docs)} chunks"
                )
            ]
        ),
    }