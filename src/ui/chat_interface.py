"""Streamlit chat interface for the SereneAI dental assistant.

Behavior preserved; improvements include docstrings and minor readability tweaks.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from typing import Any, Dict, List

import streamlit as st

from src.agents.orchestrator import run_agent
from src.tools.yolo_tool import detect_issues
import threading


@st.cache_resource
def init_db() -> sqlite3.Connection:
    """Initialize the SQLite database connection and schema if needed."""
    logging.info("UI: Initializing DB")
    conn = sqlite3.connect("dental_chatbot.db", check_same_thread=False)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS history
                 (
                     id INTEGER PRIMARY KEY,
                     user_input TEXT,
                     response TEXT,
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                 )"""
    )
    conn.commit()
    return conn


@st.cache_resource
def get_db_connection() -> sqlite3.Connection:
    """Return a cached DB connection for the Streamlit app."""
    return init_db()


@st.cache_resource
def load_rag_system():
    """Load RAG system with pre-built indices (cached across sessions).

    This function is cached by Streamlit, ensuring the RAG system is loaded
    only once per application lifetime, dramatically improving startup time.

    Returns:
        RAGSystem instance with indices loaded from disk
    """
    logging.info("RAG: Loading system (will be cached)...")
    from src.rag import RAGSystem

    rag = RAGSystem()
    rag.setup(force_rebuild=False)  # Load from disk only
    logging.info("RAG: [OK] Loaded and cached")
    return rag


@st.cache_resource
def get_rag_system_singleton():
    """Get cached RAGSystem singleton for agent use.

    This prevents agents from reinstantiating RAGSystem on every request,
    which was causing 3-5 second overhead per request.

    Returns:
        Cached RAGSystem instance
    """
    logging.info("RAG Singleton: Initializing for agent use...")
    from src.rag import RAGSystem

    rag = RAGSystem()
    rag.setup(force_rebuild=False)  # Load pre-built indices
    logging.info("RAG Singleton: [OK] Cached and ready")
    return rag


@st.cache_resource
def warmup_resources() -> bool:
    """Warm up heavy resources in a background thread without blocking UI.

    Note: RAG system is now loaded via load_rag_system() with proper caching.
    This function handles additional warmup tasks.
    """
    def _warm():
        try:
            logging.info("WARMUP: Starting background warmup")
            # Lazy imports to avoid overhead on module import
            from src.tools.yolo_tool import load_yolo_model
            from src.utils.llm import get_gemini_chat
            from langchain_core.messages import HumanMessage

            # Preload YOLO
            load_yolo_model()
            # Prewarm Gemini
            try:
                _ = get_gemini_chat(temperature=0).invoke([HumanMessage(content="ping")])
            except Exception:
                pass
            logging.info("WARMUP: Completed")
        except Exception as e:
            logging.warning(f"WARMUP: Failed {e}")

    threading.Thread(target=_warm, daemon=True).start()
    return True


def render_chat_interface(config: Dict[str, Any]) -> None:
    """Render the main chat interface and handle events.

    Args:
        config: Global runtime configuration from src.config.load_config().
    """
    logging.info("UI: Rendering chat interface")
    # Kick off background warmup
    warmup_resources()
    conn = get_db_connection()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        logging.info("UI: New session - messages initialized")

    # Replay history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"]) if message.get("content") else None
            if "image_path" in message and message["image_path"]:
                if os.path.exists(message["image_path"]):
                    try:
                        st.image(message["image_path"], width=200, caption="Uploaded Image")
                        logging.debug(
                            f"UI: Displayed historical image: {message['image_path']}"
                        )
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Could not display image: {str(e)}")
                        logging.error(f"UI: Image display error: {str(e)}")
                else:
                    st.warning("Image file no longer available")
                    logging.warning(
                        f"UI: Missing historical image: {message['image_path']}"
                    )

    prompt = st.chat_input(
        "Ask about dental issues... (or attach an image 📎)",
        accept_file=True,
        file_type=["jpg", "jpeg", "png"],
    )

    if prompt:
        logging.info(
            f"UI: New user input received - Text: '{prompt.text if hasattr(prompt, 'text') else 'No text'}', Files: {len(prompt.files if hasattr(prompt, 'files') else [])}"
        )
        user_text = prompt.text if hasattr(prompt, "text") and prompt.text else ""
        files = prompt.files if hasattr(prompt, "files") and prompt.files else []

        image_path = None
        if files:
            uploaded_file = files[0]
            unique_id = uuid.uuid4().hex
            image_path = f"temp_{unique_id}.jpg"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            logging.info(f"UI: Temp image saved: {image_path}")

        with st.chat_message("user"):
            if user_text:
                st.markdown(user_text)
            if image_path and os.path.exists(image_path):
                st.image(image_path, width=200, caption="Uploaded")

        user_dict: Dict[str, Any] = {"role": "user", "content": user_text}
        if image_path:
            user_dict["image_path"] = image_path
        st.session_state.messages.append(user_dict)
        logging.info(
            f"UI: User message appended to history (len now: {len(st.session_state.messages)})"
        )

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            try:
                with st.spinner("Processing..."):
                    logging.info(
                        f"UI: Calling run_agent - Text: '{user_text}', Image: {image_path}, History len: {len(st.session_state.messages[:-1])}"
                    )
                    full_result = run_agent(
                        user_text, image_path, st.session_state.messages[:-1]
                    )  # Dict: {'response', 'sources'}
                    response = full_result["response"]
                    sources = full_result.get("sources", [])

                # Smooth sentence streaming
                def generate_response():
                    sentences = response.split(". ")
                    for sentence in sentences:
                        if sentence:
                            yield sentence + ". "

                response_placeholder.write_stream(generate_response)
                logging.info(
                    f"UI: Response streamed (length: {len(response)} chars, sources: {len(sources)})"
                )

                # Transparansi: Tampilkan sources jika ada
                if sources:
                    # Count by provider for header
                    pdf_count = len([s for s in sources if s.get('provider') == 'PDF'])
                    pubmed_count = len([s for s in sources if s.get('provider') == 'PubMed'])
                    with st.expander("📚 Sources (RAG Context)", expanded=False):
                        st.markdown(
                            f"**Retrieved {len(sources)} sources:** {pdf_count} PDF, {pubmed_count} PubMed"
                        )
                        for src in sources:
                            provider = src.get("provider", "")
                            title = src.get("title", "Unknown")
                            snippet = src.get("snippet", "")
                            page = src.get("page", "")
                            with st.container(border=True):
                                if provider == "PubMed":
                                    pmid = src.get("pmid", "")
                                    url = src.get("url") or (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "")
                                    st.markdown(f"**Source {src['id']} (PubMed):** {title}")
                                    meta = []
                                    if src.get("authors"):
                                        meta.append(f"Authors: {src['authors']}")
                                    if pmid:
                                        meta.append(f"PMID: [ {pmid} ]({url})")
                                    if meta:
                                        st.caption(" | ".join(meta))
                                    st.caption(snippet)
                                else:
                                    # PDF source
                                    st.markdown(
                                        f"**Source {src['id']} (PDF):** {title}{f' (Page {page})' if page else ''}"
                                    )
                                    spath = src.get("source_path", "")
                                    if spath:
                                        st.caption(f"Path: {spath}")
                                    st.caption(snippet)
                        logging.debug(
                            f"UI: Displayed {len(sources)} sources in expander"
                        )
            except Exception as e:  # noqa: BLE001 - surface error in UI only
                response = f"Maaf, ada kesalahan: {str(e)}. Coba lagi."
                logging.error(f"UI: Chat processing error: {str(e)}")
                response_placeholder.markdown(response)
                sources = []

        st.session_state.messages.append(
            {"role": "assistant", "content": response, "sources": sources}
        )  # Store sources in message
        logging.debug(
            f"UI: Assistant message appended (snippet: {response[:100]}...)"
        )

        db_input = user_text if user_text else "(Image query)"
        try:
            c = conn.cursor()
            c.execute(
                "INSERT INTO history (user_input, response) VALUES (?, ?)",
                (db_input, response),
            )
            conn.commit()
            logging.debug(f"UI: Saved to DB - Input: {db_input[:50]}...")
        except Exception as e:  # noqa: BLE001 - continue without failing UI
            logging.error(f"UI: DB save error: {str(e)}")

        if image_path and os.path.exists(image_path):
            try:
                logging.info(f"UI: Processing YOLO for image: {image_path}")
                _, annotated_path, spatial_insights = detect_issues(image_path)
                if annotated_path and os.path.exists(annotated_path):
                    with st.chat_message("assistant"):
                        st.image(annotated_path, width=400, caption="Detected Image")
                        st.markdown(f"**Spatial Insights:** {spatial_insights}")
                        logging.info(
                            f"UI: Displayed annotated image & insights (snippet: {spatial_insights[:100]}...)"
                        )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"**Spatial Insights:** {spatial_insights}",
                            "image_path": annotated_path,
                        }
                    )
                # Cleanup temp files
                try:
                    os.remove(image_path)
                    if "annotated_path" in locals() and os.path.exists(annotated_path):
                        os.remove(annotated_path)
                    logging.debug(
                        f"UI: Cleaned up temp files: {image_path}, {annotated_path if 'annotated_path' in locals() else 'N/A'}"
                    )
                except Exception as ce:  # noqa: BLE001 - warn only
                    logging.warning(f"UI: Cleanup error: {ce}")
            except Exception as e:  # noqa: BLE001 - surface in UI and continue
                st.error(f"Image processing error: {str(e)}")
                logging.error(f"UI: YOLO processing error: {str(e)}")
