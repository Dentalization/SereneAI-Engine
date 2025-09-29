# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project overview
- Language: Python
- Primary entrypoint: Streamlit UI at src/ui/chat_interface.py
- Core responsibilities:
  - Conversational agent orchestration using LangGraph + Gemini (Google Generative AI)
  - Image understanding via YOLO (Ultralytics) with optional custom model
  - Retrieval-augmented generation (RAG) over local PDFs (docs/) and optional PubMed
  - Lightweight persistence of chat history in SQLite (dental_chatbot.db)

Environment and setup (Windows PowerShell)
- Create a virtual environment and install dependencies:
  - python -m venv .venv
  - .\.venv\Scripts\Activate.ps1
  - pip install -r requirements.txt
- Required environment variables (must be set before running):
  - GEMINI_API_KEY
  - Optional: COHERE_API_KEY
  Set them as environment variables without printing their values. Example:
  - $env:GEMINI_API_KEY={{GEMINI_API_KEY}}
  - $env:COHERE_API_KEY={{COHERE_API_KEY}}
- Optional local assets:
  - docs/: Place PDF files to be indexed for RAG.
  - models/oral_detection_model.pt: Custom dental YOLO model. If missing, yolo_tool falls back to a general YOLO model (yolo11n.pt).

Run the application
- Launch the Streamlit UI:
  - streamlit run src/ui/chat_interface.py
- The UI will:
  - Accept text input and optional image upload (jpg/png)
  - Stream responses from the agent
  - Persist chat history in dental_chatbot.db (created in repo root)
  - For images, run detection and visualize annotated output inline

Build, lint, and tests
- No pyproject/Makefile/tox/pytest configuration detected in this repository at the time of writing.
  - Build: Not applicable.
  - Lint: No explicit linting configuration present.
  - Tests: No tests/ directory or test runner config found.

High-level architecture
- UI (src/ui/chat_interface.py)
  - Streamlit app managing the chat experience and lightweight persistence.
  - Accepts text and optional image uploads, displays assistant responses and sources.
  - Calls run_agent(...) for conversational responses and detect_issues(...) for image analysis.
- Orchestrator (src/agents/orchestrator.py)
  - Defines an AgentState (conversation state) and LangGraph nodes:
    - orchestrator: Uses Gemini via LangChain to decide action (greet/question/rag/yolo/end) and next node.
    - validator: Validates image input.
    - detector: Runs YOLO detection + spatial insights.
    - summarizer: Runs RAG and formats the final response with sources.
  - Graph flow:
    - Entry: orchestrator
    - Conditional edges route to validator/detector/summarizer/end
    - validator -> detector -> summarizer -> END
  - Public API: run_agent(input_text, image_path=None, history=[]): returns {'response', 'sources'}
- Tools
  - YOLO tool (src/tools/yolo_tool.py)
    - Loads custom model at models/oral_detection_model.pt if available; otherwise falls back to a general YOLO model.
    - Preprocesses/validates images; filters detections by confidence threshold from config.
    - get_spatial_insights uses Gemini (vision) via LangChain to produce spatial context for detections.
    - detect_issues(...) returns serialized detections JSON, an annotated image path, and spatial insights text.
  - RAG tool (src/tools/rag_tool.py)
    - Loads PDFs from docs/ and optionally PubMed articles, splits into chunks, builds FAISS vector index.
    - Combines semantic retrieval and BM25 (if available) with cross-encoder reranking.
    - Extracts knowledge triples to build a directed knowledge graph (networkx) for relational insights.
    - query_rag(...) retrieves documents, optionally queries the knowledge graph, and returns a response plus source metadata.
- Configuration and logging (src/config.py)
  - load_config(): pulls environment variables and runtime settings (paths, thresholds, classes/colors, spatial prompt).
  - setup_logging(): configures global logging with rotating file handler (app.log).

Operational notes
- Environment variables must be present before invoking any operation that uses Gemini.
- docs/ should contain PDFs if you want local RAG context; PubMed retrieval is attempted but gracefully handled if unavailable.
- The SQLite DB (dental_chatbot.db) is created/managed by the Streamlit app in the repo root.
- GPU is used if available (torch.cuda.is_available()); otherwise falls back to CPU.
