"""
Application settings with Pydantic validation and environment variable loading.
"""

from functools import lru_cache
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration with validation.
    All settings loaded from environment variables with sensible defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==========================================================================
    # LLM Provider Configuration
    # ==========================================================================
    gemini_api_key: str = Field(..., description="Google Gemini API key (required)")
    anthropic_api_key: str | None = Field(None, description="Anthropic Claude API key (optional)")
    openai_api_key: str | None = Field(None, description="OpenAI API key (optional)")

    # Default models
    default_model: str = Field("gemini-2.5-flash", description="Default LLM model")
    fallback_model: str | None = Field("claude-sonnet-4-5-20250929", description="Fallback model")

    # ==========================================================================
    # LangSmith Observability
    # ==========================================================================
    langsmith_api_key: str | None = Field(None, description="LangSmith API key")
    langsmith_tracing: bool = Field(True, description="Enable LangSmith tracing")
    langsmith_project: str = Field("sereneai-teledentistry", description="LangSmith project name")
    langsmith_endpoint: str = Field(
        "https://api.smith.langchain.com", description="LangSmith API endpoint"
    )

    # ==========================================================================
    # Database Configuration
    # ==========================================================================
    postgres_url: str | None = Field(None, description="PostgreSQL connection URL")
    postgres_pool_size: int = Field(20, description="PostgreSQL connection pool size")
    postgres_max_overflow: int = Field(10, description="PostgreSQL max overflow connections")
    sqlite_path: str = Field(".checkpoints/dev.db", description="SQLite database path")

    # ==========================================================================
    # Security Configuration
    # ==========================================================================
    jwt_secret_key: str = Field(..., description="JWT secret key (min 32 chars)")
    jwt_algorithm: str = Field("HS256", description="JWT signing algorithm")
    jwt_expiration_minutes: int = Field(60, description="JWT token expiration (minutes)")
    aes_encryption_key: str | None = Field(None, description="AES encryption key (base64)")

    # Rate limiting
    rate_limit_requests_per_minute: int = Field(60, description="Rate limit per minute")
    rate_limit_burst: int = Field(100, description="Rate limit burst capacity")

    # ==========================================================================
    # RAG Configuration
    # ==========================================================================
    embedding_model: str = Field(
        "NeuML/pubmedbert-base-embeddings", description="HuggingFace embedding model"
    )
    rag_index_dir: str = Field(".rag/faiss_index", description="FAISS index directory")
    kg_path: str = Field(".rag/kg.pkl", description="Knowledge graph pickle path")
    rag_top_k: int = Field(10, description="Number of documents to retrieve")
    rag_similarity_threshold: float = Field(0.7, description="Similarity threshold")
    enable_pubmed: bool = Field(False, description="Enable PubMed retrieval")

    # ==========================================================================
    # Vision/YOLO Configuration
    # ==========================================================================
    yolo_model_path: str = Field(
        "models/oral_detection_model.pt", description="YOLO model path"
    )
    yolo_confidence_threshold: float = Field(0.3, description="YOLO confidence threshold")
    yolo_device: Literal["cpu", "cuda"] = Field("cpu", description="YOLO device")

    # ==========================================================================
    # Clinical Configuration
    # ==========================================================================
    language: Literal["id", "en"] = Field("id", description="Default language")
    emergency_keywords: List[str] = Field(
        default_factory=lambda: ["darurat", "emergency", "sakit_hebat", "perdarahan", "trauma"],
        description="Emergency detection keywords",
    )
    socrates_completeness_threshold: int = Field(
        5, description="SOCRATES completeness threshold"
    )
    max_conversation_turns: int = Field(50, description="Maximum conversation turns")

    # ==========================================================================
    # Application Configuration
    # ==========================================================================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        "INFO", description="Logging level"
    )
    log_file: str = Field("logs/sereneai.log", description="Log file path")
    log_rotation_size_mb: int = Field(10, description="Log rotation size (MB)")
    log_retention_days: int = Field(30, description="Log retention (days)")

    # API configuration
    api_host: str = Field("0.0.0.0", description="API host")
    api_port: int = Field(8000, description="API port")
    api_workers: int = Field(4, description="API workers")
    api_cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8501"],
        description="CORS allowed origins",
    )

    # File upload limits
    max_file_size_mb: int = Field(10, description="Max file size (MB)")
    allowed_image_formats: List[str] = Field(
        default_factory=lambda: ["jpg", "jpeg", "png"], description="Allowed image formats"
    )

    # ==========================================================================
    # Feature Flags
    # ==========================================================================
    enable_appointment_booking: bool = Field(False, description="Enable appointment booking")
    enable_medication_checker: bool = Field(False, description="Enable medication checker")
    enable_referral_system: bool = Field(False, description="Enable referral system")
    enable_differential_diagnosis: bool = Field(
        True, description="Enable differential diagnosis"
    )
    enable_treatment_planning: bool = Field(True, description="Enable treatment planning")

    # ==========================================================================
    # Middleware Configuration
    # ==========================================================================
    # PII Protection
    enable_pii_detection: bool = Field(True, description="Enable PII detection")
    pii_redaction_strategy: Literal["redact", "mask", "hash", "block"] = Field(
        "mask", description="PII redaction strategy"
    )

    # Guardrails
    enable_content_safety: bool = Field(True, description="Enable content safety")
    enable_jailbreak_detection: bool = Field(True, description="Enable jailbreak detection")
    enable_hallucination_detection: bool = Field(True, description="Enable hallucination detection")

    # Human-in-the-Loop
    enable_hitl: bool = Field(False, description="Enable human-in-the-loop")
    hitl_tools: List[str] = Field(
        default_factory=lambda: ["medication_checker", "referral"], description="HITL tools"
    )

    # ==========================================================================
    # Checkpointing & Durability
    # ==========================================================================
    checkpoint_mode: Literal["sync", "async", "exit"] = Field(
        "async", description="Checkpoint mode"
    )
    checkpoint_cleanup_days: int = Field(90, description="Checkpoint cleanup (days)")

    # ==========================================================================
    # Monitoring & Telemetry
    # ==========================================================================
    enable_prometheus: bool = Field(True, description="Enable Prometheus metrics")
    prometheus_port: int = Field(9090, description="Prometheus port")
    enable_audit_logging: bool = Field(True, description="Enable audit logging")
    audit_log_path: str = Field("logs/audit.jsonl", description="Audit log path")

    # ==========================================================================
    # Development/Testing
    # ==========================================================================
    environment: Literal["development", "staging", "production"] = Field(
        "production", description="Environment"
    )
    debug: bool = Field(False, description="Debug mode")
    reload: bool = Field(False, description="Hot reload")

    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret_key(cls, v: str) -> str:
        """Validate JWT secret key is at least 32 characters."""
        if len(v) < 32:
            raise ValueError("JWT secret key must be at least 32 characters")
        return v

    @field_validator("rag_similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, v: float) -> float:
        """Validate similarity threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        return v

    @field_validator("yolo_confidence_threshold")
    @classmethod
    def validate_confidence_threshold(cls, v: float) -> float:
        """Validate confidence threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        return v

    @property
    def use_postgres(self) -> bool:
        """Check if PostgreSQL is configured."""
        return self.postgres_url is not None

    @property
    def langsmith_enabled(self) -> bool:
        """Check if LangSmith is enabled and configured."""
        return self.langsmith_tracing and self.langsmith_api_key is not None


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to avoid re-reading environment variables on every call.
    """
    return Settings()
