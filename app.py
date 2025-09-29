# app.py
import streamlit as st
from src.ui.chat_interface import render_chat_interface
from src.config import load_config, setup_logging
from dotenv import load_dotenv
import logging

load_dotenv()
setup_logging()
config = load_config()

def main():
    """Main app with minimal UI optimizations."""
    st.set_page_config(page_title="Dental AI Chatbot", layout="wide")

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
