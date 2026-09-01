import os
import time
import uuid

import requests
import streamlit as st
import logfire

from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

env_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".env",
    )
)

load_dotenv(
    dotenv_path=env_path
)


# ============================================================
# LOGFIRE
# ============================================================

try:

    token = os.getenv(
        "LOGFIRE_TOKEN"
    )

    logfire.configure(
        token=token
    )

    LOGFIRE_STATUS = (
        "Connected & Tracing"
    )

except Exception as exc:

    LOGFIRE_STATUS = (
        f"Standby: {exc}"
    )


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# AVATARS
# ============================================================

AI_AVATAR = "🤖"
USER_AVATAR = "👤"


# ============================================================
# SESSION MEMORY
# ============================================================

if "session_id" not in st.session_state:

    st.session_state.session_id = (
        str(uuid.uuid4())
    )

    logfire.info(
        "✨ New chat session created | "
        f"id={st.session_state.session_id}"
    )


if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🧠 Agent OS"
    )

    st.markdown("---")

    st.success(
        f"Logfire: {LOGFIRE_STATUS}"
    )

    st.info(
        "Memory ID: "
        f"{st.session_state.session_id[:8]}"
    )

    if st.button(
        "🗑️ Clear History & Memory",
        width="stretch",
        type="primary",
    ):

        old_session = (
            st.session_state.session_id
        )

        logfire.info(
            "🗑️ Clearing chat session | "
            f"id={old_session}"
        )

        st.session_state.messages = []

        st.session_state.session_id = (
            str(uuid.uuid4())
        )

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🤖 Enterprise Agentic Assistant"
)


st.caption(
    "Documentation-grounded assistant with "
    "session memory and strict guardrails."
)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    avatar = (
        AI_AVATAR
        if message["role"] == "assistant"
        else USER_AVATAR
    )

    with st.chat_message(
        message["role"],
        avatar=avatar,
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

if prompt := st.chat_input(
    "Ask about your documentation..."
):

    # ========================================================
    # USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )


    with st.chat_message(
        "user",
        avatar=USER_AVATAR,
    ):

        st.markdown(
            prompt
        )


    # ========================================================
    # ASSISTANT
    # ========================================================

    with st.chat_message(
        "assistant",
        avatar=AI_AVATAR,
    ):

        with st.status(
            "🔍 Agent is thinking...",
            expanded=True,
        ) as status:

            try:

                # ============================================
                # BACKEND
                # ============================================

                base_url = os.getenv(
                    "BACKEND_URL",
                    "http://localhost:8000",
                )

                url = (
                    f"{base_url}/query"
                )


                # ============================================
                # IMPORTANT:
                #
                # SAME thread_id for the entire chat.
                # ============================================

                payload = {
                    "q": prompt,
                    "thread_id": (
                        st.session_state.session_id
                    ),
                }


                logfire.info(
                    "📡 Calling backend | "
                    f"thread={st.session_state.session_id}"
                )


                response = requests.post(
                    url,
                    json=payload,
                    timeout=60,
                )


                response.raise_for_status()


                data = response.json()


                # ============================================
                # THOUGHT PROCESS
                # ============================================

                steps = data.get(
                    "thought_process",
                    [],
                )


                for step in steps:

                    st.write(
                        f"⚙️ {step}"
                    )


                # ============================================
                # STATUS
                # ============================================

                backend_status = data.get(
                    "status",
                    "",
                )


                status.update(
                    label="✅ Request processed",
                    state="complete",
                    expanded=False,
                )


                # ============================================
                # SOURCES
                # ============================================

                sources = data.get(
                    "sources",
                    [],
                )


                if sources:

                    with st.expander(
                        "📄 View Retrieved Context"
                    ):

                        for i, source in enumerate(
                            sources
                        ):

                            preview = (
                                str(source)[:100]
                                .replace(
                                    "\n",
                                    " ",
                                )
                                + "..."
                            )

                            with st.expander(
                                f"Chunk {i + 1}: {preview}"
                            ):

                                st.info(
                                    source
                                )


            except requests.exceptions.RequestException as exc:

                logfire.error(
                    "❌ Backend connection failed: "
                    f"{exc}"
                )

                status.update(
                    label="❌ Backend connection failed",
                    state="error",
                )

                st.error(
                    "Backend is unavailable."
                )

                st.stop()


            except Exception as exc:

                logfire.error(
                    "❌ UI processing failed: "
                    f"{exc}"
                )

                status.update(
                    label="❌ Request failed",
                    state="error",
                )

                st.error(
                    "Something went wrong while processing "
                    "the request."
                )

                st.stop()


        # ====================================================
        # ANSWER
        # ====================================================

        full_answer = data.get(
            "answer",
            "No response.",
        )


        # ====================================================
        # STREAMING EFFECT
        # ====================================================

        answer_placeholder = st.empty()

        current_text = ""


        for char in full_answer:

            current_text += char

            answer_placeholder.markdown(
                current_text + "▌"
            )

            time.sleep(
                0.005
            )


        answer_placeholder.markdown(
            full_answer
        )


        # ====================================================
        # SAVE UI HISTORY
        # ====================================================

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_answer,
            }
        )


        logfire.info(
            "✅ Chat cycle completed | "
            f"thread={st.session_state.session_id}"
        )