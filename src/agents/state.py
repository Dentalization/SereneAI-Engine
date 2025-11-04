"""
State models for teledentistry agent using LangChain 1.0 patterns.
Extends AgentState from LangChain for compatibility with create_agent().
"""

from datetime import datetime
from typing import Annotated, List, Literal, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field


# =============================================================================
# Clinical Models (Pydantic for validation)
# =============================================================================


class SOCRATESProfile(BaseModel):
    """
    SOCRATES framework for structured symptom assessment.
    Medical standard for pain/symptom evaluation.
    """

    site: str | None = Field(None, description="Location of dental problem")
    onset: str | None = Field(None, description="When symptoms started")
    character: str | None = Field(None, description="Pain type (sharp, dull, throbbing)")
    radiation: str | None = Field(None, description="Pain spreading pattern")
    associations: List[str] = Field(
        default_factory=list, description="Related symptoms (swelling, bleeding, fever)"
    )
    time_course: str | None = Field(None, description="Symptom progression over time")
    exacerbating_factors: List[str] = Field(
        default_factory=list, description="What makes symptoms worse"
    )
    relieving_factors: List[str] = Field(
        default_factory=list, description="What makes symptoms better"
    )
    severity: int | None = Field(None, ge=0, le=10, description="Pain scale 0-10")

    def completeness_score(self) -> int:
        """Calculate how many SOCRATES elements are filled."""
        score = 0
        if self.site:
            score += 1
        if self.onset:
            score += 1
        if self.character:
            score += 1
        if self.radiation:
            score += 1
        if self.associations:
            score += 1
        if self.time_course:
            score += 1
        if self.exacerbating_factors:
            score += 1
        if self.relieving_factors:
            score += 1
        if self.severity is not None:
            score += 1
        return score

    def is_complete(self, threshold: int = 5) -> bool:
        """Check if SOCRATES is sufficiently complete."""
        return self.completeness_score() >= threshold or (self.severity or 0) >= 7


class DetectionResult(BaseModel):
    """Dental pathology detection result from YOLO."""

    class_name: str = Field(..., description="Detected dental condition")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    spatial_description: str | None = Field(None, description="Location description")


class SourceCitation(BaseModel):
    """Evidence source with provenance tracking."""

    title: str = Field(..., description="Document title")
    provider: str = Field(..., description="Source provider (PDF, PubMed, etc.)")
    snippet: str = Field(..., description="Relevant excerpt")
    confidence: float = Field(..., ge=0, le=1, description="Relevance confidence")
    page_number: int | None = Field(None, description="Page number if applicable")
    url: str | None = Field(None, description="URL if available")
    authors: List[str] = Field(default_factory=list, description="Document authors")


class UserProfile(BaseModel):
    """Patient profile with medical history."""

    # Identity
    user_id: str | None = Field(None, description="Unique user identifier")
    name: str | None = Field(None, description="Patient name")
    age: int | None = Field(None, ge=0, le=150, description="Patient age")
    gender: Literal["male", "female", "other"] | None = Field(None, description="Gender")

    # Preferences
    language: Literal["id", "en"] = Field("id", description="Preferred language")

    # Medical History
    medical_conditions: List[str] = Field(
        default_factory=list, description="Pre-existing medical conditions"
    )
    medications: List[str] = Field(default_factory=list, description="Current medications")
    allergies: List[str] = Field(default_factory=list, description="Known allergies")
    dental_history: List[str] = Field(
        default_factory=list, description="Previous dental treatments"
    )

    # Current Consultation
    symptoms: SOCRATESProfile = Field(
        default_factory=SOCRATESProfile, description="Current symptoms (SOCRATES)"
    )
    chief_complaint: str | None = Field(None, description="Main concern in patient's words")

    # Detection History
    detections: List[DetectionResult] = Field(
        default_factory=list, description="Detected dental conditions"
    )

    def merge_symptoms(self, new_symptoms: SOCRATESProfile) -> None:
        """
        Deep merge new symptoms with existing ones.
        Preserves non-null values, accumulates lists.
        """
        if new_symptoms.site:
            self.symptoms.site = new_symptoms.site
        if new_symptoms.onset:
            self.symptoms.onset = new_symptoms.onset
        if new_symptoms.character:
            self.symptoms.character = new_symptoms.character
        if new_symptoms.radiation:
            self.symptoms.radiation = new_symptoms.radiation
        if new_symptoms.associations:
            # Merge lists, remove duplicates
            self.symptoms.associations = list(
                set(self.symptoms.associations + new_symptoms.associations)
            )
        if new_symptoms.time_course:
            self.symptoms.time_course = new_symptoms.time_course
        if new_symptoms.exacerbating_factors:
            self.symptoms.exacerbating_factors = list(
                set(self.symptoms.exacerbating_factors + new_symptoms.exacerbating_factors)
            )
        if new_symptoms.relieving_factors:
            self.symptoms.relieving_factors = list(
                set(self.symptoms.relieving_factors + new_symptoms.relieving_factors)
            )
        if new_symptoms.severity is not None:
            self.symptoms.severity = new_symptoms.severity


# =============================================================================
# Agent State (TypedDict for LangChain compatibility)
# =============================================================================


class TeledentistryState(TypedDict):
    """
    Main state for teledentistry agent.
    Extends LangChain's AgentState pattern with custom fields.

    Note: Must be TypedDict, not Pydantic, for LangChain 1.0 compatibility.
    Pydantic models are no longer supported for state schemas.
    """

    # Core LangChain fields
    messages: Annotated[List[AnyMessage], "Conversation messages with append reducer"]

    # User profile (persistent across conversation)
    user_profile: NotRequired[UserProfile]

    # Conversation metadata
    conversation_id: NotRequired[str]
    created_at: NotRequired[str]  # ISO timestamp
    updated_at: NotRequired[str]  # ISO timestamp
    language: NotRequired[Literal["id", "en"]]

    # Routing & execution
    next_action: NotRequired[str]  # Next action to take
    stage: NotRequired[str]  # Conversation stage
    confidence: NotRequired[float]  # Overall confidence

    # Clinical outputs
    diagnosis: NotRequired[List[dict]]  # Differential diagnosis
    treatment_plan: NotRequired[List[dict]]  # Treatment recommendations
    sources: NotRequired[List[SourceCitation]]  # Evidence sources
    emergency_detected: NotRequired[bool]  # Emergency flag

    # Metadata for middleware
    pii_detected: NotRequired[bool]  # PII detection flag
    guardrails_triggered: NotRequired[List[str]]  # Triggered guardrails
    tool_calls_count: NotRequired[int]  # Number of tool calls
    execution_time_ms: NotRequired[float]  # Execution time


# =============================================================================
# Conversation Stage Enum
# =============================================================================


class ConversationStage:
    """Conversation lifecycle stages."""

    GREETING = "greeting"
    ANAMNESIS = "anamnesis"  # Symptom gathering
    DIAGNOSIS = "diagnosis"  # Analysis & differential diagnosis
    TREATMENT_PLAN = "treatment_plan"  # Treatment recommendations
    REFERRAL = "referral"  # Dentist referral needed
    FOLLOW_UP = "follow_up"  # Follow-up questions
    CLOSURE = "closure"  # End conversation


# =============================================================================
# Context Object (for ToolRuntime)
# =============================================================================


class ConsultationContext(BaseModel):
    """
    Immutable context passed to tools via ToolRuntime.
    Contains configuration and user-specific settings.
    """

    user_id: str
    language: Literal["id", "en"]
    conversation_id: str
    thread_id: str

    # Feature flags
    enable_differential_diagnosis: bool = True
    enable_treatment_planning: bool = True
    enable_appointment_booking: bool = False
    enable_medication_checker: bool = False
    enable_referral_system: bool = False

    # Clinical settings
    socrates_threshold: int = 5
    emergency_keywords: List[str] = Field(default_factory=list)

    # RAG settings
    rag_top_k: int = 10
    rag_similarity_threshold: float = 0.7

    # Vision settings
    yolo_confidence_threshold: float = 0.3
