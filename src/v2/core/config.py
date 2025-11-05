"""
Configuration Management for SereneAI V2
Centralized configuration with environment variable support
"""

import os
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


# ============================================================================
# BASE PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / ".checkpoints_v2"


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

class ModelConfig(BaseModel):
    """LLM model configuration"""

    # Primary model for most agents
    primary_model: str = "gemini-2.5-flash"

    # Vision-capable model
    vision_model: str = "gemini-2.0-flash-exp"

    # Embedding model
    embedding_model: str = "NeuML/pubmedbert-base-embeddings"

    # Model parameters
    temperature: float = 0.1
    max_tokens: int = 2048
    top_p: float = 0.95

    # Rate limiting
    max_requests_per_minute: int = 60
    max_tokens_per_minute: int = 40000


# ============================================================================
# YOLO CONFIGURATION
# ============================================================================

class YOLOConfig(BaseModel):
    """YOLO detection configuration"""

    model_path: Path = MODELS_DIR / "oral_detection_model.pt"
    fallback_model: str = "yolo11n.pt"
    confidence_threshold: float = 0.3
    iou_threshold: float = 0.45
    device: Literal["cpu", "cuda", "mps"] = "cuda"

    # Detection classes
    classes: list[str] = Field(default_factory=lambda: [
        "calculus",
        "caries",
        "gingivitis",
        "hypodontia",
        "tooth_discoloration",
        "ulcer"
    ])

    # Image preprocessing
    input_size: int = 640
    max_image_size_mb: int = 5
    min_resolution: tuple[int, int] = (100, 100)
    supported_formats: list[str] = Field(default_factory=lambda: ["jpg", "jpeg", "png"])


# ============================================================================
# RAG CONFIGURATION
# ============================================================================

class RAGConfig(BaseModel):
    """RAG system configuration"""

    # Index paths
    faiss_index_dir: Path = DATA_DIR / "faiss_index_v2"
    knowledge_graph_path: Path = DATA_DIR / "kg_v2.pkl"
    metadata_path: Path = DATA_DIR / "index_metadata_v2.json"

    # Document sources
    docs_dir: Path = DATA_DIR / "docs"
    enable_pubmed: bool = False

    # Retrieval parameters
    top_k: int = 5
    similarity_threshold: float = 0.7
    rerank_top_n: int = 3

    # Chunking parameters
    chunk_size: int = 500
    chunk_overlap: int = 100
    semantic_similarity_threshold: float = 0.75

    # Query expansion
    expand_queries: bool = True
    max_expansions: int = 3

    # Validation
    enable_claim_validation: bool = True
    validation_threshold: float = 0.8


# ============================================================================
# PERSISTENCE CONFIGURATION
# ============================================================================

class PersistenceConfig(BaseModel):
    """Persistence and memory configuration"""

    # PostgreSQL checkpointer
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "sereneai_v2"
    postgres_user: str = "sereneai"
    postgres_password: str = "changeme"

    # Redis store for long-term memory
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None

    # Checkpoint configuration
    checkpoint_ttl_days: int = 30
    enable_compression: bool = True

    # Memory configuration
    enable_long_term_memory: bool = True
    memory_search_top_k: int = 5
    memory_similarity_threshold: float = 0.8

    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL"""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """Get Redis connection URL"""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"


# ============================================================================
# API CONFIGURATION
# ============================================================================

class APIConfig(BaseModel):
    """FastAPI server configuration"""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 4

    # CORS
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # Security
    enable_auth: bool = False
    api_key_header: str = "X-API-Key"
    rate_limit_per_minute: int = 60

    # Streaming
    enable_streaming: bool = True
    stream_mode: list[str] = Field(default_factory=lambda: ["updates", "messages"])


# ============================================================================
# OBSERVABILITY CONFIGURATION
# ============================================================================

class ObservabilityConfig(BaseModel):
    """Observability and monitoring configuration"""

    # LangSmith
    enable_langsmith: bool = True
    langsmith_project: str = "sereneai-v2"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: Path = LOGS_DIR
    log_rotation: str = "1 day"
    log_retention: str = "30 days"

    # Metrics
    enable_metrics: bool = True
    metrics_port: int = 9090


# ============================================================================
# MIDDLEWARE CONFIGURATION
# ============================================================================

class MiddlewareConfig(BaseModel):
    """Middleware configuration"""

    # PII Detection
    enable_pii_detection: bool = True
    pii_redaction_strategy: Literal["redact", "mask", "hash", "block"] = "redact"

    # Summarization
    enable_summarization: bool = True
    summarization_threshold: int = 4000  # tokens
    summary_length: int = 500

    # Human-in-the-loop
    enable_human_approval: bool = False
    approval_required_for: list[str] = Field(default_factory=lambda: [
        "emergency_referral",
        "medication_recommendation"
    ])

    # Guardrails
    enable_guardrails: bool = True
    blocked_keywords: list[str] = Field(default_factory=list)


# ============================================================================
# MAIN SETTINGS
# ============================================================================

class Settings(BaseSettings):
    """Main application settings loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Environment
    environment: Literal["development", "staging", "production"] = "development"

    # API Keys
    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")
    cohere_api_key: str | None = Field(None, alias="COHERE_API_KEY")
    langsmith_api_key: str | None = Field(None, alias="LANGSMITH_API_KEY")

    # Component configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    yolo: YOLOConfig = Field(default_factory=YOLOConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    middleware: MiddlewareConfig = Field(default_factory=MiddlewareConfig)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SINGLETON
# ============================================================================

@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
