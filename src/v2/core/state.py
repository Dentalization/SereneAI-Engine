"""
State Management for SereneAI V2
LangChain 1.0 compliant state schemas using TypedDict

Based on LangChain 1.0 best practices:
- Use TypedDict (NOT Pydantic BaseModel for state)
- Extend langchain.agents.AgentState
- Define reducers for list/dict merging
"""

from typing import TypedDict, Annotated, Sequence, Literal
from typing_extensions import NotRequired
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


# ============================================================================
# MESSAGE TYPES
# ============================================================================

class MessageContent(TypedDict):
    """Structured message content"""
    type: Literal["text", "image", "tool_result"]
    text: NotRequired[str]
    image_url: NotRequired[str]
    tool_name: NotRequired[str]
    tool_result: NotRequired[dict]


# ============================================================================
# USER PROFILE & MEDICAL DATA
# ============================================================================

class SOCRATESProfile(TypedDict, total=False):
    """SOCRATES symptom assessment framework"""
    site: str  # Location of symptoms
    onset: str  # When did it start
    character: str  # Nature of symptoms (sharp, dull, aching)
    radiation: str  # Does pain spread elsewhere
    associations: list[str]  # Associated symptoms (swelling, bleeding, etc.)
    time_course: str  # Pattern over time (constant, intermittent, progressive)
    exacerbating_factors: list[str]  # What makes it worse
    severity: int  # 1-10 scale


class MedicalHistory(TypedDict, total=False):
    """Patient medical history"""
    conditions: list[str]  # Existing medical conditions
    medications: list[str]  # Current medications
    allergies: list[str]  # Known allergies
    previous_dental: list[str]  # Previous dental procedures


class Demographics(TypedDict, total=False):
    """Patient demographics"""
    age: int
    gender: Literal["male", "female", "other", "prefer_not_to_say"]
    language: Literal["id", "en"]  # Indonesian or English
    location: str


class UserProfile(TypedDict, total=False):
    """Complete user profile - accumulates across conversations"""
    demographics: Demographics
    medical_history: MedicalHistory
    symptoms: SOCRATESProfile
    preferences: dict[str, str]  # UI preferences, notification settings, etc.


# ============================================================================
# DETECTION & ANALYSIS
# ============================================================================

class DetectionResult(TypedDict):
    """YOLO detection result"""
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    severity_estimate: NotRequired[str]  # AI-assessed severity


class VisionAnalysis(TypedDict):
    """Vision analysis results"""
    detections: list[DetectionResult]
    spatial_insights: str  # Gemini Vision spatial analysis
    quality_score: float  # Image quality (0-1)
    quality_issues: list[str]  # Issues affecting analysis
    recommended_actions: list[str]


# ============================================================================
# RAG & KNOWLEDGE
# ============================================================================

class SourceCitation(TypedDict):
    """RAG source citation"""
    title: str
    content: str
    source_type: Literal["guideline", "textbook", "pubmed", "clinical_trial"]
    url: NotRequired[str]
    relevance_score: float
    page: NotRequired[int]


class ValidationResult(TypedDict):
    """Claim validation result"""
    is_supported: bool
    confidence: float
    supporting_sources: list[str]
    contradicting_sources: list[str]
    explanation: str


# ============================================================================
# AGENT DECISION & ROUTING
# ============================================================================

class TriageDecision(TypedDict):
    """Triage agent decision"""
    intent: Literal[
        "medical_query",
        "symptom_assessment",
        "image_analysis",
        "general_info",
        "appointment_request",
        "emergency"
    ]
    confidence: float
    requires_image: bool
    requires_anamnesis: bool
    requires_rag: bool
    next_agent: Literal[
        "anamnesis",
        "vision",
        "rag",
        "synthesis",
        "human_handoff"
    ]
    reasoning: str


# ============================================================================
# CONVERSATION STAGE
# ============================================================================

ConversationStage = Literal[
    "greeting",
    "triage",
    "anamnesis",
    "image_collection",
    "analysis",
    "diagnosis_support",
    "education",
    "referral",
    "completed"
]


# ============================================================================
# MAIN AGENT STATE
# ============================================================================

def merge_user_profile(existing: UserProfile | None, new: UserProfile) -> UserProfile:
    """
    Deep merge user profiles - preserves existing data, only updates new fields
    Special handling for SOCRATES symptoms and list fields
    """
    if existing is None:
        return new

    result = dict(existing)

    for key, value in new.items():
        if value is None:
            continue  # Don't overwrite with None

        if key not in result:
            result[key] = value
        elif isinstance(value, dict):
            # Deep merge nested dicts
            result[key] = {**result[key], **value}  # type: ignore
        elif isinstance(value, list):
            # Deduplicate and merge lists
            existing_list = result.get(key, [])
            if isinstance(existing_list, list):
                result[key] = list(set(existing_list) | set(value))  # type: ignore
        else:
            # Scalar values - new overwrites old
            result[key] = value

    return result  # type: ignore


class AgentState(TypedDict):
    """
    Main state for SereneAI agent graph

    IMPORTANT: This must be TypedDict (NOT Pydantic BaseModel) for LangChain 1.0
    """

    # ============================================================================
    # CONVERSATION MANAGEMENT
    # ============================================================================

    # Messages channel - uses add_messages reducer for automatic merging
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Conversation identifiers
    conversation_id: str
    thread_id: str  # For checkpointing
    user_id: NotRequired[str]  # For multi-tenant

    # Current stage in conversation
    stage: ConversationStage

    # ============================================================================
    # USER DATA
    # ============================================================================

    # User profile - uses custom reducer for deep merge
    user_profile: Annotated[UserProfile, merge_user_profile]

    # ============================================================================
    # AGENT OUTPUTS
    # ============================================================================

    # Triage
    triage_decision: NotRequired[TriageDecision]

    # Anamnesis
    anamnesis_complete: bool
    anamnesis_data: NotRequired[dict]  # Extracted symptom data

    # Vision
    vision_analysis: NotRequired[VisionAnalysis]
    image_path: NotRequired[str]

    # RAG
    rag_response: NotRequired[str]
    sources: NotRequired[list[SourceCitation]]
    validation: NotRequired[ValidationResult]

    # Synthesis
    final_response: NotRequired[str]

    # ============================================================================
    # ROUTING & CONTROL FLOW
    # ============================================================================

    # Next node to route to (for conditional edges)
    next_node: NotRequired[Literal[
        "triage",
        "anamnesis",
        "vision",
        "rag",
        "synthesis",
        "human_approval",
        "emergency_handoff",
        "__end__"
    ]]

    # ============================================================================
    # METADATA
    # ============================================================================

    # Overall confidence in final response
    confidence_score: NotRequired[float]

    # Execution metadata
    execution_time_ms: NotRequired[int]
    model_calls: NotRequired[int]
    tokens_used: NotRequired[int]

    # Error tracking
    errors: NotRequired[list[str]]
    warnings: NotRequired[list[str]]

    # ============================================================================
    # HUMAN-IN-THE-LOOP
    # ============================================================================

    # Approval required for certain actions
    requires_approval: NotRequired[bool]
    approval_reason: NotRequired[str]
    approved_actions: NotRequired[list[str]]
    rejected_actions: NotRequired[list[str]]


# ============================================================================
# RUNTIME CONTEXT (immutable per-request)
# ============================================================================

class RuntimeContext(TypedDict):
    """
    Immutable context passed to tools and middleware
    Does NOT go in state - passed via config
    """

    # User identifiers
    user_id: str
    session_id: str

    # API configuration
    api_keys: dict[str, str]

    # Feature flags
    enable_pubmed: bool
    enable_vision: bool
    enable_human_approval: bool

    # Limits
    max_message_length: int
    max_image_size_mb: int
    max_tool_calls: int

    # Observability
    trace_id: str
    langsmith_project: str
