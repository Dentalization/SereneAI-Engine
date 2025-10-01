"""Enhanced state models using Pydantic for validation and persistence.

This module defines the conversation state and related models with:
- Strong typing and validation via Pydantic
- Conversation history tracking
- User profile management
- Source citation models
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ConversationStage(str, Enum):
    """Stages of the dental consultation conversation."""

    GREETING = "greeting"
    ANAMNESIS = "anamnesis"
    DIAGNOSIS = "diagnosis"
    TREATMENT_PLAN = "treatment_plan"
    REFERRAL = "referral"
    CLOSURE = "closure"


class MessageRole(str, Enum):
    """Message roles in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Individual message in conversation history."""

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    image_path: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SourceCitation(BaseModel):
    """Citation source from RAG retrieval."""

    id: int
    title: str
    provider: str  # "PDF" or "PubMed"
    snippet: str
    source_path: Optional[str] = None
    url: Optional[str] = None
    page: Optional[int] = None
    authors: Optional[str] = None
    pmid: Optional[str] = None
    confidence: float = 1.0


class SOCRATESProfile(BaseModel):
    """Structured symptom profile using SOCRATES framework.

    SOCRATES:
    - Site: Location of pain/issue
    - Onset: When did it start
    - Character: Nature of pain (sharp, dull, throbbing)
    - Radiation: Does it spread
    - Associations: Other symptoms
    - Time course: Progression
    - Exacerbating/Relieving: What makes it better/worse
    - Severity: Pain scale 1-10
    """

    site: Optional[str] = Field(None, description="Location of dental issue")
    onset: Optional[str] = Field(None, description="When symptoms started")
    character: Optional[str] = Field(None, description="Nature of pain/symptoms")
    radiation: Optional[str] = Field(None, description="Does pain spread elsewhere")
    associations: List[str] = Field(default_factory=list, description="Associated symptoms")
    time_course: Optional[str] = Field(None, description="How symptoms changed over time")
    exacerbating_factors: List[str] = Field(default_factory=list, description="What makes it worse")
    relieving_factors: List[str] = Field(default_factory=list, description="What makes it better")
    severity: Optional[int] = Field(None, ge=1, le=10, description="Pain severity 1-10")

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v):
        if v is not None and not (1 <= v <= 10):
            raise ValueError("Severity must be between 1 and 10")
        return v

    @classmethod
    def from_dict(cls, data: Dict) -> 'SOCRATESProfile':
        """Safe deserialization from dict.

        Args:
            data: Dict or SOCRATESProfile instance

        Returns:
            SOCRATESProfile instance
        """
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            # Filter only valid fields
            valid_fields = {k: v for k, v in data.items() if k in cls.model_fields}
            return cls(**valid_fields)
        return cls()  # Return empty if invalid


class UserProfile(BaseModel):
    """User profile with medical history and demographics."""

    user_id: str = Field(default="anonymous")
    language: str = Field(default="id", description="Preferred language code")
    age: Optional[int] = None
    gender: Optional[str] = None

    # Medical history
    medical_conditions: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    previous_dental_work: List[str] = Field(default_factory=list)

    # Current consultation
    symptoms: SOCRATESProfile = Field(default_factory=SOCRATESProfile)
    chief_complaint: Optional[str] = None

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DetectionResult(BaseModel):
    """YOLO detection result for a single finding."""

    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    spatial_description: Optional[str] = None


class AgentState(BaseModel):
    """Enhanced conversation state passed between LangGraph nodes."""

    # Core input/output
    conversation_id: str = Field(default_factory=lambda: f"conv_{int(datetime.now().timestamp())}")
    input: str = Field(default="")
    final_response: str = Field(default="")

    # Image analysis
    image_path: Optional[str] = None
    detections: List[DetectionResult] = Field(default_factory=list)
    spatial_insights: str = Field(default="")

    # RAG results
    rag_response: str = Field(default="")
    sources: List[SourceCitation] = Field(default_factory=list)

    # Conversation management
    history: List[ChatMessage] = Field(default_factory=list)
    conversation_stage: ConversationStage = ConversationStage.GREETING
    user_profile: UserProfile = Field(default_factory=UserProfile)

    # Agent orchestration
    triage_decision: Optional[Dict[str, Any]] = None
    anamnesis_data: Optional[Dict[str, Any]] = None
    confidence_score: float = 1.0

    # Routing
    next_node: str = Field(default="end")

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True

    def add_message(self, role: MessageRole, content: str, **metadata) -> None:
        """Add a message to conversation history."""
        self.history.append(
            ChatMessage(role=role, content=content, metadata=metadata)
        )

    def get_history_string(self, last_n: int = 5) -> str:
        """Get formatted conversation history for prompts."""
        recent = self.history[-last_n:] if last_n else self.history
        return "\n".join([f"{msg.role.value}: {msg.content}" for msg in recent])

    def update_profile(self, **kwargs) -> None:
        """Update user profile fields with safe serialization."""
        for key, value in kwargs.items():
            if hasattr(self.user_profile, key):
                # Special handling for symptoms/SOCRATES profile
                if key == "symptoms" and isinstance(value, dict):
                    value = SOCRATESProfile.from_dict(value)
                setattr(self.user_profile, key, value)
        self.user_profile.last_updated = datetime.now()


class CheckpointState(BaseModel):
    """Persistent checkpoint for conversation state."""

    conversation_id: str
    state: AgentState
    timestamp: datetime = Field(default_factory=datetime.now)
    version: int = 1

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "conversation_id": self.conversation_id,
            "state": self.state.model_dump(mode='json'),
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CheckpointState:
        """Deserialize from dictionary."""
        return cls(
            conversation_id=data["conversation_id"],
            state=AgentState(**data["state"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            version=data.get("version", 1),
        )