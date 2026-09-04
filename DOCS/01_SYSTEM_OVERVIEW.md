# 🤖 Enterprise Agentic RAG: System Overview

A production-grade RAG system built for speed, scalability, and observability. This platform uses **LangGraph** for planner/retriever/responder orchestration, local document parsing and embeddings, Qdrant search, and Portkey-backed generation.

---

## 🌟 Vision
Most RAG systems fail because they treat every query the same. Our **Agentic RAG** distinguishes between:
1.  **Conversational Queries**: "Hi", "Who are you?", "What did I just say?"
2.  **Technical Queries**: "How do I configure Intel SRIOV on Kubernetes?"

By using a **Planner-Retriever-Responder** architecture, we ensure that technical answers are always grounded in "True Data" while conversational interactions remain fluid and fast.

---

## 🏗️ High-Level Flow
```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit UI
    participant Agent as Agent Brain (FastAPI)
    participant Data as Knowledge Base (Qdrant)

    User->>UI: Asks Question
    UI->>Agent: Request with thread_id
    Agent->>Agent: Planner decides intent
    alt Technical
        Agent->>Data: Embed query + vector search
        Data-->>Agent: Raw Chunks
        Agent->>Agent: FlashRank Local Reranking
    else Conversational
        Agent->>Agent: Recall Memory
    end
    Agent->>User: Synthesized Answer + Sources
```

---

## 📂 Project Organization
*   **`app/`**: The core Python package containing the Agent, Pipelines, and Services.
*   **`ui/`**: A Streamlit chat interface that displays the answer, execution steps, and retrieved chunk text.
*   **`DATA/`**: The ground-truth documentation used for ingestion.
*   **`DOCS/`**: This documentation suite.
*   **`commands.md`**: The master execution guide for developers.

---

## 🚀 Quick Navigation
1.  **Ingestion**: [02_INGESTION_ENGINE.md](02_INGESTION_ENGINE.md)
2.  **Intelligence**: [03_NODE_INTELLIGENCE.md](03_NODE_INTELLIGENCE.md)
3.  **Observability**: [04_TRACING_AND_OBSERVABILITY.md](04_TRACING_AND_OBSERVABILITY.md)
4.  **Environment Variables**: [05_ENVIRONMENT_VARIABLES.md](05_ENVIRONMENT_VARIABLES.md)
