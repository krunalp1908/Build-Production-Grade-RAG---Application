# ⚠️ Known Gotchas & Architectural Decisions

This document tracks non-obvious platform quirks and explains why the
architecture is designed the way it is. It is a **living reference** — it
grows one entry at a time as each lesson introduces the code that the
gotcha applies to.

---

## 1. Embedding Dimension Is Resolved at Runtime, Not Hardcoded

**The Issue:**
`app/services/retrieval/embedding.py` uses the local
`sentence-transformers/all-mpnet-base-v2` model. Its vector dimension is
resolved from the loaded model rather than hardcoded, so the Qdrant
collection matches the active model.

**The Solution:**
`app/ingestion/processor.py` calls `get_embedding_dim()` — which actually
probes the local embedding model immediately before creating the Qdrant
collection, so the collection's vector size matches the vectors written to it.

---

## 2. Chunking Is Character-Length + Paragraph-Boundary Only

**The Issue:**
`app/ingestion/chunking/splitter.py`'s `chunk_text()` splits on blank lines
(`"\n\n"`) and greedily accumulates paragraphs up to a 1500-character
ceiling — it does not do token-aware splitting and does not add overlap
between chunks. This is intentional for this stage: it's simple, fast, and
good enough for the document types in `DATA/true_data/`. Keep it in mind if
you swap in documents with very long, unbroken paragraphs.

---

## 3. Logfire Initialization Order (The "Poisoning" Bug)

**The Issue:**
If any module calls `logfire.info()`/`logfire.span()` *before*
`logfire.configure()` has run, Logfire's internal state becomes "poisoned"
for that process — it silently enters a no-op mode and discards all
subsequent traces, even if you call `.configure()` later. Importing
`app.config.settings` (or anything that transitively imports it) at the top
of `app/main.py` risks pulling in a module with a module-level Logfire call
before configuration happens.

**The Solution:**
`app/main.py` bypasses `app.config` entirely at the very top of the file —
it loads `.env` and calls `logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))`
using raw `os.getenv()`, before importing anything else from `app`.

```python
# app/main.py
import logfire
import os
from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# Safe to import the rest of the application now!
from app.agents.graph import rag_agent
```

## 4. Conversation Memory Requires a Stable `thread_id`

**The Issue:**
`app/agents/graph.py` now compiles the `StateGraph` with a `MemorySaver`
checkpointer, and `/query` passes `config={"configurable": {"thread_id": ...}}`
on every call. `MemorySaver` keys its history purely by `thread_id` — if the
client generates a new one per request (or doesn't send one at all, falling
back to the `"default_user"` default on every caller), every call looks like
a brand-new conversation even though the checkpointer is working correctly.

**The Solution:**
The Streamlit UI (`ui/app.py`) generates one `thread_id` per browser session
(via `uuid.uuid4()`) and reuses it for every message in that session — it's
the caller's job to keep sending the same `thread_id`, not the graph's.

---

## 5. Reranking Has a Graceful Fallback

**The Issue:**
`app/services/retrieval/ranking_service.py` lazily loads a FlashRank
`Ranker` on first use. If model loading fails for any reason (e.g. no disk
space for the ONNX model cache), the whole `/query` request would otherwise
fail even though the Qdrant search itself succeeded.

**The Solution:**
`rerank_documents()` catches that failure and falls back to returning
`documents[:top_n]` in their original Qdrant-ranked order — reranking
degrades gracefully instead of taking down the whole pipeline.

---

## 6. The Colang File Is Not the Active Request Gate

**The Issue:**
`app/guardrails/colang_rules.py` contains historical Colang/YAML policy
material, but the live `/query` path calls the Python `guard()` function.

**The Solution:**
`initialize_rails()` creates a direct `ChatGroq` classifier using
`GROQ_GUARD_MODEL`. `guard()` applies deterministic regex/dialog checks and
then asks that classifier for exactly `ALLOW` or `BLOCK`.

---

## 7. Guardrails Is Input-Only, Never Output-Gated

**The Issue:**
It's easy to assume a "guardrails" layer checks both what goes in and what
comes out. This one doesn't: `guard(q)` in `app/main.py` is called exactly
once, on the raw user message, before `rag_agent.invoke(...)`. The LLM's
`final_answer` is returned to the client with no guardrails check at all.

**Why it's left this way:** input gating catches the cases this course
demonstrates (off-topic questions, jailbreak attempts) without the added
latency and complexity of a second guardrails pass on every response.
Output-gating is a reasonable extension, just not one this project
implements.

---

## 8. Two Different Ways to Call Through the Gateway — On Purpose

**The Issue:**
`app/gateway/client.py` exposes both a native `portkey_client` (Portkey's
own SDK) and `get_langchain_llm()` (a `ChatOpenAI` wrapper pointed at
Portkey's OpenAI-compatible endpoint). It would be easy to assume this is
inconsistent and standardize on one.

**Why both exist:** `app/agents/nodes/planner.py` just needs a plain
`.invoke(prompt).content` call, so it uses the LangChain wrapper for a
familiar interface. `app/agents/nodes/responder.py` needs to read the
`x-portkey-cache-status` response header to know whether a request was
served from cache — LangChain's `ChatOpenAI` wrapper doesn't expose that
header, so `responder.py` uses the native Portkey client instead. Both
routes use the saved `PORTKEY_CONFIG_ID`, so fallback, cache, and retry
behavior are controlled centrally by Portkey.

---

## 9. Golden Dataset Sample Paths Are Case-Sensitive

**The Issue:**
`evals/data_parser.py` builds its sample corpus by reading from
`DATA/true_data/` and `DATA/noisy_data/` directly (not through
`app/ingestion/processor.py`). On Windows, filesystem lookups are
case-insensitive, so a directory reference typo (`data/` vs `DATA/`) fails
silently — until the same code runs on a case-sensitive filesystem (Mac,
Linux, most CI runners), where it raises `FileNotFoundError`. Keep casing
exact when referencing `DATA/` anywhere in this repo.
