"""Multi-agent orchestrator using LangGraph.

Features:
- Specialized agents (Triage, Anamnesis, Vision, RAG, Synthesis)
- Robust error handling with circuit breakers and fallbacks
- Conversation state persistence
- Enhanced state management with Pydantic models
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from src.agents.persistence import save_state
from src.agents.specialized.anamnesis_agent import AnamnesisAgent
from src.agents.specialized.rag_agent import RAGAgent
from src.agents.specialized.synthesis_agent import SynthesisAgent
from src.agents.specialized.triage_agent import TriageAgent
from src.agents.specialized.vision_agent import VisionAgent
from src.agents.state_models import AgentState, ConversationStage, MessageRole

logger = logging.getLogger(__name__)

# Initialize specialized agents
triage_agent = TriageAgent()
anamnesis_agent = AnamnesisAgent()
vision_agent = VisionAgent()
rag_agent = RAGAgent()
synthesis_agent = SynthesisAgent()


def triage_node(state: AgentState) -> Dict[str, Any]:
    """Triage node: Classify query and route to appropriate path."""
    logger.info(f"Orchestrator: Triage node - Input: '{state.input[:50]}...'")

    try:
        result = triage_agent.execute(state=state)

        if result.status.value != "success":
            logger.error(f"Orchestrator: Triage failed - {result.error}")
            # Fallback: conservative routing
            return {
                "conversation_stage": ConversationStage.ANAMNESIS,
                "next_node": "anamnesis" if not state.image_path else "vision",
                "confidence_score": 0.5,
            }

        # Extract decision
        decision_data = result.data
        state.conversation_stage = decision_data["stage"]
        state.confidence_score = decision_data["confidence"]
        state.triage_decision = decision_data

        # Update profile if needed
        if decision_data.get("profile_update"):
            state.update_profile(**decision_data["profile_update"])

        # Handle direct responses (greet, question)
        action = decision_data["action"]
        if action in ["greet", "question"] and decision_data.get("response"):
            state.final_response = decision_data["response"]
            state.next_node = "end"
            logger.info(f"Orchestrator: Direct response - {action}")
            return {"final_response": state.final_response, "next_node": "end"}

        # Route based on action
        next_node = {
            "yolo": "vision",
            "rag": "rag",
            "question": "anamnesis",
        }.get(action, "end")

        state.next_node = next_node
        logger.info(f"Orchestrator: Triage decision - stage={state.conversation_stage}, next={next_node}")

        return {
            "conversation_stage": state.conversation_stage,
            "next_node": next_node,
            "confidence_score": state.confidence_score,
            "triage_decision": decision_data,
        }

    except Exception as e:
        logger.error(f"Orchestrator: Triage node exception - {e}")
        return {
            "final_response": "Maaf, terjadi kesalahan. Bisakah Anda ulangi pertanyaan?",
            "next_node": "end",
        }


def anamnesis_node(state: AgentState) -> Dict[str, Any]:
    """Anamnesis node: Extract structured symptoms using SOCRATES."""
    logger.info("Orchestrator: Anamnesis node")

    try:
        result = anamnesis_agent.execute(state=state)

        if result.status.value != "success":
            logger.warning(f"Orchestrator: Anamnesis extraction failed - {result.error}")
            # Continue with partial data
            return {
                "final_response": "Bisakah Anda ceritakan lebih detail tentang keluhan Anda?",
                "next_node": "end",
            }

        anamnesis_data = result.data
        state.anamnesis_data = anamnesis_data

        # Update profile with symptoms
        if anamnesis_data.get("profile_update"):
            state.update_profile(**anamnesis_data["profile_update"])

        # Check if ready for diagnosis
        if anamnesis_data.get("ready_for_diagnosis"):
            logger.info("Orchestrator: Sufficient info collected, routing to RAG")
            state.next_node = "rag"
            return {
                "anamnesis_data": anamnesis_data,
                "next_node": "rag",
            }
        else:
            # Ask follow-up question
            suggested_q = anamnesis_data.get("suggested_question", "Ceritakan lebih lanjut?")
            logger.info("Orchestrator: More info needed, asking follow-up")
            state.final_response = suggested_q
            state.next_node = "end"
            return {
                "final_response": suggested_q,
                "anamnesis_data": anamnesis_data,
                "next_node": "end",
            }

    except Exception as e:
        logger.error(f"Orchestrator: Anamnesis node exception - {e}")
        return {
            "final_response": "Untuk membantu lebih baik, bisa ceritakan: di mana sakit, sejak kapan, dan seberapa parah (1-10)?",
            "next_node": "end",
        }


def vision_node(state: AgentState) -> Dict[str, Any]:
    """Vision node: Analyze dental image with YOLO."""
    logger.info("Orchestrator: Vision node")

    if not state.image_path:
        logger.warning("Orchestrator: Vision node called without image, skipping")
        return {"next_node": "rag"}

    try:
        result = vision_agent.execute(state=state)

        if result.status.value != "success":
            logger.error(f"Orchestrator: Vision analysis failed - {result.error}")
            state.spatial_insights = "Image analysis failed. Please try with a clearer image."
            return {
                "spatial_insights": state.spatial_insights,
                "next_node": "rag",
            }

        vision_data = result.data

        # Check image quality
        if not vision_data.get("success", False):
            # Image quality issues
            quality = vision_data.get("image_quality")
            if quality:
                feedback = (
                    f"Image quality needs improvement. "
                    f"Issues: {', '.join(quality.issues)}. "
                    f"Suggestions: {', '.join(quality.recommendations)}"
                )
                state.final_response = feedback
                state.next_node = "end"
                return {
                    "final_response": feedback,
                    "next_node": "end",
                }

        # Extract detections and spatial insights
        state.detections = vision_data.get("detections", [])
        state.spatial_insights = vision_data.get("spatial_insights", "")
        state.confidence_score = vision_data.get("confidence", 0.7)

        logger.info(f"Orchestrator: Vision complete - {len(state.detections)} detections")

        # Continue to RAG for evidence-based advice
        state.next_node = "rag"
        return {
            "detections": state.detections,
            "spatial_insights": state.spatial_insights,
            "next_node": "rag",
            "confidence_score": state.confidence_score,
        }

    except Exception as e:
        logger.error(f"Orchestrator: Vision node exception - {e}")
        return {
            "spatial_insights": "Image processing error occurred.",
            "next_node": "rag",
        }


def rag_node(state: AgentState) -> Dict[str, Any]:
    """RAG node: Retrieve evidence and validate response."""
    logger.info("Orchestrator: RAG node")

    try:
        result = rag_agent.execute(state=state)

        if result.status.value != "success":
            logger.error(f"Orchestrator: RAG failed - {result.error}")
            # Fallback response
            state.final_response = (
                "Berdasarkan keluhan Anda, saya sarankan untuk segera berkonsultasi dengan "
                "dokter gigi untuk pemeriksaan lebih lanjut dan penanganan yang tepat."
            )
            state.next_node = "synthesis"
            return {
                "rag_response": state.final_response,
                "next_node": "synthesis",
            }

        rag_data = result.data

        state.rag_response = rag_data.get("response", "")
        state.sources = rag_data.get("sources", [])
        state.confidence_score = rag_data.get("overall_confidence", 0.7)

        logger.info(
            f"Orchestrator: RAG complete - Confidence={state.confidence_score:.2f}, "
            f"Sources={len(state.sources)}, Risk={rag_data.get('hallucination_risk', 'unknown')}"
        )

        # Pass recommendations to synthesis
        state.next_node = "synthesis"
        return {
            "rag_response": state.rag_response,
            "sources": state.sources,
            "confidence_score": state.confidence_score,
            "next_node": "synthesis",
            "recommendations": rag_data.get("recommendations", []),
            "claim_validations": rag_data.get("claim_validations", []),
        }

    except Exception as e:
        logger.error(f"Orchestrator: RAG node exception - {e}")
        return {
            "rag_response": "Untuk keluhan Anda, konsultasi dokter gigi sangat disarankan.",
            "next_node": "synthesis",
        }


def synthesis_node(state: AgentState) -> Dict[str, Any]:
    """Synthesis node: Assemble final response with citations."""
    logger.info("Orchestrator: Synthesis node")

    try:
        # Collect all data for synthesis
        synthesis_kwargs = {
            "rag_response": state.rag_response,
            "sources": state.sources,
            "recommendations": state.triage_decision.get("recommendations", []) if state.triage_decision else [],
            "overall_confidence": state.confidence_score,
        }

        result = synthesis_agent.execute(state=state, **synthesis_kwargs)

        if result.status.value != "success":
            logger.warning(f"Orchestrator: Synthesis failed - {result.error}, using RAG response directly")
            state.final_response = state.rag_response or "Konsultasi dokter gigi direkomendasikan."
        else:
            synthesis_data = result.data
            state.final_response = synthesis_data.get("final_response", state.rag_response)

        # Add message to history
        state.add_message(MessageRole.ASSISTANT, state.final_response)

        # Save checkpoint
        save_state(state)

        logger.info(f"Orchestrator: Synthesis complete - Response length: {len(state.final_response)}")

        return {
            "final_response": state.final_response,
            "next_node": "end",
        }

    except Exception as e:
        logger.error(f"Orchestrator: Synthesis node exception - {e}")
        return {
            "final_response": state.rag_response or "Maaf, terjadi kesalahan. Konsultasi dokter gigi disarankan.",
            "next_node": "end",
        }


# Build LangGraph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("triage", triage_node)
graph.add_node("anamnesis", anamnesis_node)
graph.add_node("vision", vision_node)
graph.add_node("rag", rag_node)
graph.add_node("synthesis", synthesis_node)

# Set entry point
graph.set_entry_point("triage")

# Add conditional routing from triage
graph.add_conditional_edges(
    "triage",
    lambda s: s.next_node,
    {
        "anamnesis": "anamnesis",
        "vision": "vision",
        "rag": "rag",
        "end": END,
    },
)

# Anamnesis can go to RAG or END (more questions)
graph.add_conditional_edges(
    "anamnesis",
    lambda s: s.next_node,
    {
        "rag": "rag",
        "end": END,
    },
)

# Vision always goes to RAG (for evidence-based interpretation)
graph.add_edge("vision", "rag")

# RAG goes to synthesis
graph.add_edge("rag", "synthesis")

# Synthesis is terminal
graph.add_edge("synthesis", END)

# Compile graph
app = graph.compile()


def run_agent(
    input_text: str,
    image_path: str | None = None,
    history: list[dict] | None = None,
    conversation_id: str | None = None,
) -> Dict[str, Any]:
    """Run multi-agent orchestration for dental chatbot.

    Args:
        input_text: User input text
        image_path: Optional path to dental image
        history: Prior conversation messages
        conversation_id: Optional ID to resume conversation

    Returns:
        Dict with 'response', 'sources', 'confidence', 'conversation_id'
    """
    logger.info(f"Orchestrator: Starting run - Input: '{input_text[:50]}...', Image: {bool(image_path)}")

    # Initialize state
    from src.agents.state_models import ChatMessage

    state = AgentState(
        input=input_text,
        image_path=image_path or "",
        conversation_id=conversation_id or f"conv_{int(__import__('time').time())}",
    )

    # Restore history if provided
    if history:
        for msg in history:
            role = MessageRole(msg.get("role", "user"))
            content = msg.get("content", "")
            state.add_message(role, content)

    try:
        # Run graph
        result = app.invoke(state)

        response = result.final_response or "Maaf, terjadi kesalahan."
        sources = result.sources or []

        logger.info(
            f"Orchestrator: Run complete - Response: {len(response)} chars, "
            f"Sources: {len(sources)}, Confidence: {result.confidence_score:.2f}"
        )

        return {
            "response": response,
            "sources": [src.model_dump() if hasattr(src, 'model_dump') else src for src in sources],
            "confidence": result.confidence_score,
            "conversation_id": result.conversation_id,
        }

    except Exception as e:
        logger.error(f"Orchestrator: Run failed - {e}")
        return {
            "response": "Maaf, ada kesalahan teknis. Coba lagi atau konsultasi dokter gigi.",
            "sources": [],
            "confidence": 0.0,
            "conversation_id": state.conversation_id,
        }