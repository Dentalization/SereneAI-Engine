# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SereneAI Engine is a production-ready dental AI chatbot system built with a multi-agent architecture using LangGraph. The system integrates computer vision (YOLO), retrieval-augmented generation (RAG), and specialized conversational agents to provide dental consultations.

**Core Technologies**: Python, Streamlit, LangGraph, LangChain, YOLO, FAISS, Gemini API

## Essential Commands

### Setup and Dependencies

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies (no requirements.txt - dependencies managed manually)
# Key packages: streamlit, langchain, ultralytics, faiss-cpu, sentence-transformers
```

### RAG System

```bash
# Build RAG indices (REQUIRED before first run)
python scripts/build_indices.py

# Build with PubMed integration
python scripts/build_indices.py --enable-pubmed

# Custom docs path
python scripts/build_indices.py --docs-path /path/to/docs
```

**Important**: The RAG system uses a two-phase architecture:
- **Ingestion Phase (offline)**: `scripts/build_indices.py` - Builds FAISS index and knowledge graph, saves to `.rag/`
- **Runtime Phase (online)**: `app.py` loads pre-built indices from disk for fast startup

### Running the Application

```bash
# Start Streamlit app
streamlit run app.py

# With custom log level
LOG_LEVEL=DEBUG streamlit run app.py
```

### Development

No testing framework is currently configured. The codebase does not have unit tests.

## Architecture

### Multi-Agent Orchestration

The system uses **LangGraph** for orchestrating specialized agents. All agents inherit from `BaseAgent` (src/agents/specialized/base_agent.py), which provides:
- Retry logic with exponential backoff
- Circuit breaker pattern for fault tolerance
- Fallback chain support

**Agent Flow**:
1. **Triage Agent** - Classifies query, routes to appropriate agent
2. **Anamnesis Agent** - Extracts structured symptoms using SOCRATES framework
3. **Vision Agent** - Processes dental images with YOLO detection
4. **RAG Agent** - Retrieves information from knowledge base
5. **Synthesis Agent** - Generates final response with citations

**Key File**: `src/agents/orchestrator.py` - Defines the LangGraph state machine

### State Management

Conversation state is managed using **Pydantic models** in `src/agents/state_models.py`:
- `AgentState` - Main conversation state with history, user profile, routing info
- `SOCRATESProfile` - Structured symptom profile (Site, Onset, Character, Radiation, Associations, Time course, Exacerbating/Relieving factors, Severity)
- `ChatMessage` - Individual messages with role, content, timestamp
- `SourceCitation` - RAG retrieval citations

**Persistence**: Conversation states are checkpointed to `.checkpoints/` directory (see `src/agents/persistence.py`)

### RAG System Architecture

The RAG system (src/rag/rag_system.py) has a **two-phase design**:

**Ingestion (Offline - scripts/build_indices.py)**:
1. Load documents (PDFs + optional PubMed)
2. Semantic chunking (src/rag/semantic_chunker.py)
3. Generate embeddings
4. Build FAISS vectorstore → save to `.rag/faiss_index/`
5. Build knowledge graph → save to `.rag/kg.pkl`

**Runtime (Online - app.py)**:
1. Load pre-built FAISS index from disk
2. Load knowledge graph from disk
3. Lazy-load heavy components (reranker, claim validator)

**Key Components**:
- `QueryExpander` - Expands queries with medical synonyms
- `AdvancedReranker` - ColBERT-style reranking with temporal scoring
- `KnowledgeGraphBuilder` - Builds entity-relationship graph from documents
- `ClaimValidator` - Validates claims and traces citations to prevent hallucinations

### Vision Pipeline

**YOLO-based Detection** (src/tools/yolo_tool.py):
- Custom dental model: `models/oral_detection_model.pt`
- Detects 6 classes: calculus, caries, gingivitis, hypodontia, tooth_discoloration, ulcer
- Fallback to `yolo11n.pt` if custom model fails
- GPU acceleration if available

**Spatial Analysis**:
- Gemini Vision API adds spatial context (upper/lower jaw, left/right, tooth position)
- Caching of spatial analysis results (LRU cache, max 64 entries)

**Async Pipeline** (src/vision/async_yolo.py):
- Concurrent processing of multiple images
- Background preprocessing

### LLM Integration

**Primary**: Gemini API (via LangChain)
**Fallback Chain** (src/utils/fallback_llm.py):
1. Gemini Flash (fast)
2. Gemini Pro (balanced)
3. Simple template-based response (last resort)

**Utility**: `src/utils/llm.py` - Centralized LLM client creation

### Configuration

All configuration is in `src/config.py`:
- Loads environment variables from `.env`
- Returns dict with model paths, API keys, thresholds, prompts
- Use `load_config()` to access configuration
- Use `setup_logging()` to configure logging (writes to `app.log` with rotation)

**Required Environment Variables**:
```
GEMINI_API_KEY=your_key_here
```

**Optional**:
```
COHERE_API_KEY=optional
LOG_LEVEL=INFO
ENABLE_PUBMED=false
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_INDEX_DIR=.rag/faiss_index
KG_PATH=.rag/kg.pkl
```

## Directory Structure

```
src/
├── agents/              # Multi-agent orchestration
│   ├── orchestrator.py      # LangGraph state machine
│   ├── state_models.py      # Pydantic state models
│   ├── persistence.py       # Checkpoint system
│   └── specialized/         # Specialized agents
│       ├── base_agent.py        # Base class with retry/circuit breaker
│       ├── triage_agent.py      # Query classification
│       ├── anamnesis_agent.py   # SOCRATES extraction
│       ├── vision_agent.py      # Image analysis
│       ├── rag_agent.py         # Knowledge retrieval
│       └── synthesis_agent.py   # Response generation
├── rag/                 # RAG system components
│   ├── rag_system.py           # Main RAG coordinator
│   ├── semantic_chunker.py     # Threshold-based chunking
│   ├── query_expander.py       # Query expansion
│   ├── advanced_reranker.py    # ColBERT-style reranking
│   ├── knowledge_graph.py      # Entity-relationship graph
│   └── claim_validator.py      # Hallucination prevention
├── tools/               # Detection tools
│   └── yolo_tool.py            # YOLO detection + spatial analysis
├── vision/              # Async vision pipeline
│   ├── async_yolo.py           # Concurrent processing
│   └── preprocessing.py        # Image preprocessing
├── ui/                  # Streamlit interface
│   └── chat_interface.py       # Chat UI + DB persistence
├── utils/               # Shared utilities
│   ├── llm.py                  # LLM client creation
│   ├── fallback_llm.py         # Fallback chain
│   └── embeddings.py           # Centralized embeddings
└── config.py            # Configuration and logging

scripts/
└── build_indices.py     # RAG ingestion pipeline

models/
└── oral_detection_model.pt  # Custom YOLO dental model

docs/                    # PDF documents for RAG (user-provided)
.rag/                    # RAG indices (generated by build_indices.py)
.checkpoints/            # Conversation checkpoints (auto-generated)
```

## Key Patterns

### Adding a New Agent

1. Create new agent class in `src/agents/specialized/`
2. Inherit from `BaseAgent`
3. Implement `_execute(**kwargs) -> Dict[str, Any]` method
4. Add agent node to `orchestrator.py`
5. Update routing logic in `triage_node()` or other nodes

### Modifying RAG Behavior

**For ingestion changes**: Edit `scripts/build_indices.py` or `src/rag/` components, then rebuild:
```bash
python scripts/build_indices.py
```

**For runtime changes**: Edit `src/rag/rag_system.py` query methods

### Error Handling

All agents use structured error handling:
- Retry with exponential backoff (configurable via `max_retries`, `retry_delay`)
- Circuit breaker prevents cascading failures (threshold=5, timeout=60s)
- Fallback agents for graceful degradation
- Always return `AgentResult` with status, data, error, execution_time_ms

### State Updates

Never mutate state directly. Return dicts from agent nodes:
```python
def my_node(state: AgentState) -> Dict[str, Any]:
    # Process state
    return {
        "field_to_update": new_value,
        "next_node": "next_node_name"
    }
```

## Logging

- Configured via `setup_logging()` in `src/config.py`
- Writes to `app.log` (rotating, 1MB max, 5 backups)
- UTF-8 encoding to prevent Windows unicode errors
- Set level via `LOG_LEVEL` environment variable (DEBUG, INFO, WARNING, ERROR)
- Use module-level loggers: `logger = logging.getLogger(__name__)`

## Performance Optimizations

1. **RAG Startup**: Pre-built indices loaded from disk (~1s vs ~60s cold build)
2. **Streamlit Caching**: `@st.cache_resource` for RAG system, DB, YOLO model
3. **Lazy Loading**: Heavy components loaded on-demand
4. **Background Warmup**: Optional background thread for additional resources
5. **LRU Caching**: Spatial analysis results cached (64 entries)
6. **Embedding Model Sharing**: Single instance shared across RAG components

## Important Notes

- **RAG indices must be built before first run**: `python scripts/build_indices.py`
- **Custom YOLO model required**: Place `oral_detection_model.pt` in `models/` directory
- **No tests configured**: Test framework not currently implemented
- **Windows compatibility**: UTF-8 logging configured, use Windows path separators
- **Database**: SQLite DB (`dental_chatbot.db`) created automatically for chat history
- **API Keys**: Gemini API key is required; Cohere is optional
