# app.py
"""SereneAI Dental Chatbot - Main Application Entry Point.

Optimized Architecture:
- Streamlit caching for RAG system (fast startup)
- Pre-built indices loaded from disk (no runtime indexing)
- Lazy loading of heavy components
- Background warmup for additional resources

Performance:
- Cold start: ~5 seconds (first run)
- Warm start: <1 second (subsequent runs)
"""
import streamlit as st
from src.ui.chat_interface import render_chat_interface, load_rag_system
from src.config import load_config, setup_logging
from dotenv import load_dotenv
import logging

load_dotenv()
setup_logging()
config = load_config()


def main():
    """Main app with optimized resource loading."""
    st.set_page_config(
        page_title="Dental AI Chatbot",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Pre-load RAG system (cached - runs only once)
    try:
        rag = load_rag_system()
        logging.info(f"App: RAG system ready (vectorstore loaded)")
    except Exception as e:
        logging.error(f"App: Failed to load RAG system - {e}")
        st.error(
            "⚠️ RAG indices not found. "
            "Please run: `python scripts/build_indices.py` first."
        )
        st.stop()

    # Hide sidebar for minimal design
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] { display: none !important; }
    .stButton > button { background-color: var(--primary-color); color: white; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    :root {
        --primary-color: #007bff;
        --secondary-color: #28a745;
        --background-color: #f8f9fa;
        --text-color: #212529;
    }
    body {
        font-family: 'Arial', sans-serif;
        background-color: var(--background-color);
        color: var(--text-color);
    }
    </style>
    """, unsafe_allow_html=True)

    render_chat_interface(config)

if __name__ == "__main__":
    main()
