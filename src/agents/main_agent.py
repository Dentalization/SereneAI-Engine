"""
Main teledentistry agent using LangChain 1.0 create_agent().
This is the production-ready agent implementation following all 2025 best practices.
"""

from typing import Callable

from langchain.agents import create_agent

from src.agents.checkpoints import get_checkpointer
from src.agents.middleware import (
    ContextEngineeringMiddleware,
    GuardrailsMiddleware,
    ObservabilityMiddleware,
    PIIProtectionMiddleware,
)
from src.agents.state import ConsultationContext, TeledentistryState
from src.config import get_settings
from src.tools import dental_vision_analysis, rag_retrieval


def get_dynamic_system_prompt(state: TeledentistryState) -> str:
    """
    Generate dynamic system prompt based on conversation state.

    Follows LangChain 1.0 context engineering pattern:
    state-driven prompts that adapt to conversation stage.

    Args:
        state: Current agent state

    Returns:
        Contextualized system prompt
    """
    config = get_settings()
    user_profile = state.get("user_profile")
    language = user_profile.language if user_profile else config.language

    # Base prompt
    if language == "id":
        base_prompt = """Anda adalah asisten AI teledentistry profesional yang membantu pasien dengan konsultasi gigi jarak jauh.

**PERAN ANDA:**
- Memberikan informasi dental berbasis bukti ilmiah
- Mengumpulkan riwayat gejala menggunakan framework SOCRATES
- Menganalisis gambar gigi untuk deteksi kondisi
- Memberikan rekomendasi berdasarkan panduan klinis
- Merujuk ke dokter gigi profesional saat diperlukan

**PENTING - BATASAN:**
- Anda BUKAN dokter gigi dan tidak dapat memberikan diagnosis medis resmi
- Selalu rujuk ke dokter gigi untuk diagnosis dan perawatan definitif
- Jangan memberikan resep obat atau dosis medis
- Informasi Anda bersifat edukatif dan informatif, bukan pengganti konsultasi langsung

**KUALITAS RESPONS:**
- Gunakan bahasa Indonesia yang jelas dan ramah
- Berikan penjelasan dengan detail yang tepat
- Sertakan sumber rujukan untuk klaim medis
- Tanyakan pertanyaan klarifikasi jika informasi kurang lengkap

**KEAMANAN PASIEN:**
- Deteksi kondisi darurat (nyeri hebat, perdarahan, trauma)
- Rekomendasikan kunjungan mendesak jika diperlukan
- Jaga privasi dan kerahasiaan informasi pasien"""
    else:
        base_prompt = """You are a professional teledentistry AI assistant helping patients with remote dental consultations.

**YOUR ROLE:**
- Provide evidence-based dental information
- Gather symptom history using SOCRATES framework
- Analyze dental images for condition detection
- Provide recommendations based on clinical guidelines
- Refer to professional dentists when necessary

**IMPORTANT - LIMITATIONS:**
- You are NOT a dentist and cannot provide official medical diagnoses
- Always refer to a dentist for definitive diagnosis and treatment
- Do not prescribe medications or dosages
- Your information is educational and informative, not a replacement for direct consultation

**RESPONSE QUALITY:**
- Use clear and friendly language
- Provide explanations with appropriate detail
- Include source references for medical claims
- Ask clarifying questions if information is incomplete

**PATIENT SAFETY:**
- Detect emergency conditions (severe pain, bleeding, trauma)
- Recommend urgent visits when necessary
- Maintain privacy and confidentiality of patient information"""

    # Add stage-specific guidance
    stage = state.get("stage")
    if stage == "anamnesis":
        base_prompt += (
            "\n\n**CURRENT STAGE: Symptom Gathering**\n"
            "Focus on collecting complete SOCRATES information. "
            "Ask specific follow-up questions for missing elements."
        )
    elif stage == "diagnosis":
        base_prompt += (
            "\n\n**CURRENT STAGE: Analysis**\n"
            "Analyze gathered information and detected conditions. "
            "Provide differential diagnosis with confidence levels."
        )
    elif stage == "treatment_plan":
        base_prompt += (
            "\n\n**CURRENT STAGE: Recommendations**\n"
            "Provide evidence-based treatment recommendations. "
            "Include home care, when to see dentist, and preventive measures."
        )

    # Add emergency detection if high severity
    if user_profile and user_profile.symptoms.severity and user_profile.symptoms.severity >= 8:
        base_prompt += (
            "\n\n⚠️ **HIGH SEVERITY ALERT**\n"
            "Patient reports high pain level. "
            "Prioritize emergency assessment and urgent dentist referral."
        )

    return base_prompt


def create_teledentistry_agent(config: get_settings | None = None):
    """
    Create production-ready teledentistry agent using LangChain 1.0 create_agent().

    This implementation follows all 2025 best practices:
    - Official checkpointer (PostgreSQL/SQLite)
    - Middleware stack (PII, guardrails, context engineering, observability)
    - Tools with @tool decorator and ToolRuntime
    - Dynamic system prompts
    - Structured state with TypedDict
    - LangSmith observability

    Args:
        config: Settings instance (default: from get_settings())

    Returns:
        Compiled agent ready for invocation

    Example:
        >>> agent = create_teledentistry_agent()
        >>> result = agent.invoke(
        ...     {"messages": [{"role": "user", "content": "Gigi saya sakit"}]},
        ...     config={"configurable": {"thread_id": "conv_123"}}
        ... )
    """
    if config is None:
        config = get_settings()

    # Initialize tools
    tools = [
        dental_vision_analysis,
        rag_retrieval,
    ]

    # TODO: Add conditional tools based on feature flags
    # if config.enable_appointment_booking:
    #     tools.append(appointment_tool)
    # if config.enable_medication_checker:
    #     tools.append(medication_checker_tool)
    # if config.enable_referral_system:
    #     tools.append(referral_tool)

    # Configure middleware stack
    middleware = []

    # 1. Observability (first - trace everything)
    if config.langsmith_enabled:
        middleware.append(ObservabilityMiddleware())

    # 2. PII Protection (before LLM sees data)
    if config.enable_pii_detection:
        middleware.append(PIIProtectionMiddleware(strategy=config.pii_redaction_strategy))

    # 3. Guardrails (content safety, jailbreak detection)
    if config.enable_content_safety:
        middleware.append(GuardrailsMiddleware())

    # 4. Context Engineering (dynamic prompts & tool selection)
    middleware.append(ContextEngineeringMiddleware())

    # TODO: Add Human-in-the-Loop middleware
    # if config.enable_hitl:
    #     from src.agents.middleware.hitl import HumanInTheLoopMiddleware
    #     middleware.append(
    #         HumanInTheLoopMiddleware(
    #             interrupt_on={tool: True for tool in config.hitl_tools}
    #         )
    #     )

    # Get checkpointer
    checkpointer = get_checkpointer()

    # Create agent using LangChain 1.0
    agent = create_agent(
        model=config.default_model,
        tools=tools,
        system_prompt=get_dynamic_system_prompt,  # Dynamic prompt function
        middleware=middleware,
        checkpointer=checkpointer,
        state_schema=TeledentistryState,  # Extended state
        # response_format=None,  # Let tools handle structured outputs
    )

    return agent


def invoke_agent(
    user_input: str,
    conversation_id: str | None = None,
    user_profile: dict | None = None,
    image_path: str | None = None,
) -> dict:
    """
    High-level function to invoke agent with user input.

    This is the main entry point for agent invocation,
    providing a simple interface for external callers.

    Args:
        user_input: User's message
        conversation_id: Conversation ID for persistence (generates if None)
        user_profile: User profile dict (optional)
        image_path: Path to dental image (optional)

    Returns:
        Dictionary with response, sources, confidence, etc.

    Example:
        >>> result = invoke_agent(
        ...     "Gigi saya sakit di bagian kiri atas",
        ...     conversation_id="conv_123"
        ... )
        >>> print(result["response"])
    """
    import uuid
    from datetime import datetime

    from langchain_core.messages import HumanMessage

    from src.agents.state import UserProfile

    # Generate conversation ID if not provided
    if not conversation_id:
        conversation_id = f"conv_{uuid.uuid4().hex[:16]}"

    # Initialize user profile
    if user_profile:
        profile = UserProfile(**user_profile)
    else:
        profile = UserProfile(user_id=f"user_{uuid.uuid4().hex[:8]}")

    # Build message
    if image_path:
        message = HumanMessage(
            content=[
                {"type": "text", "text": user_input},
                {"type": "image_url", "image_url": {"url": image_path}},
            ]
        )
    else:
        message = HumanMessage(content=user_input)

    # Build state
    state = {
        "messages": [message],
        "user_profile": profile,
        "conversation_id": conversation_id,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "language": profile.language,
    }

    # Build context
    config_settings = get_settings()
    context = ConsultationContext(
        user_id=profile.user_id,
        language=profile.language,
        conversation_id=conversation_id,
        thread_id=conversation_id,
        enable_differential_diagnosis=config_settings.enable_differential_diagnosis,
        enable_treatment_planning=config_settings.enable_treatment_planning,
        enable_appointment_booking=config_settings.enable_appointment_booking,
        enable_medication_checker=config_settings.enable_medication_checker,
        enable_referral_system=config_settings.enable_referral_system,
        socrates_threshold=config_settings.socrates_completeness_threshold,
        emergency_keywords=config_settings.emergency_keywords,
        rag_top_k=config_settings.rag_top_k,
        rag_similarity_threshold=config_settings.rag_similarity_threshold,
        yolo_confidence_threshold=config_settings.yolo_confidence_threshold,
    )

    # Create and invoke agent
    agent = create_teledentistry_agent()
    result = agent.invoke(
        state,
        config={
            "configurable": {"thread_id": conversation_id},
            "context": context.model_dump(),
        },
    )

    # Extract response
    last_message = result["messages"][-1]
    response_text = last_message.content if hasattr(last_message, "content") else str(last_message)

    return {
        "response": response_text,
        "conversation_id": conversation_id,
        "sources": result.get("sources", []),
        "confidence": result.get("confidence", 0.0),
        "next_action": result.get("next_action"),
        "user_profile": result.get("user_profile"),
    }
