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

    result = triage_agent.execute(state=state)

    if result.status.value != "success":
        logger.error(f"Orchestrator: Triage failed - {result.error}")
        # Raise exception - NO fallback routing
        from src.utils.exceptions import TriageError
        raise TriageError(
            message=f"Triage agent failed: {result.error}",
            user_action="Could not understand your query. Please try rephrasing or provide more details about your dental concern."
        )

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


def anamnesis_node(state: AgentState) -> Dict[str, Any]:
    """Anamnesis node: Extract structured symptoms using SOCRATES."""
    logger.info("Orchestrator: Anamnesis node")

    result = anamnesis_agent.execute(state=state)

    if result.status.value != "success":
        logger.error(f"Orchestrator: Anamnesis extraction failed - {result.error}")
        # Raise exception - NO fallback
        from src.utils.exceptions import AnamnesisError
        raise AnamnesisError(
            message=f"Anamnesis agent failed: {result.error}",
            user_action="Could not extract symptom information. Please provide more details about your dental concern."
        )

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
        suggested_q = anamnesis_data.get("suggested_question")
        if not suggested_q:
            from src.utils.exceptions import AnamnesisError
            raise AnamnesisError(
                message="Anamnesis agent did not provide follow-up question",
                user_action="Could not generate follow-up question. Please provide more details."
            )

        logger.info("Orchestrator: More info needed, asking follow-up")
        state.final_response = suggested_q
        state.next_node = "end"
        return {
            "final_response": suggested_q,
            "anamnesis_data": anamnesis_data,
            "next_node": "end",
        }


def vision_node(state: AgentState) -> Dict[str, Any]:
    """Vision node: Analyze dental image with YOLO."""
    logger.info("Orchestrator: Vision node")

    if not state.image_path:
        logger.error("Orchestrator: Vision node called without image")
        from src.utils.exceptions import VisionError
        raise VisionError(
            message="Vision node called without image path",
            user_action="No image was provided. Please upload an image of your dental concern."
        )

    result = vision_agent.execute(state=state)

    if result.status.value != "success":
        logger.error(f"Orchestrator: Vision analysis failed - {result.error}")
        # Raise exception - NO fallback
        from src.utils.exceptions import VisionError
        raise VisionError(
            message=f"Vision agent failed: {result.error}",
            user_action="Could not analyze the image. Please try with a clearer dental image."
        )

    vision_data = result.data

    # Check image quality
    if not vision_data.get("success", False):
        # Image quality issues - provide specific feedback
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
        else:
            # No quality info - raise exception
            from src.utils.exceptions import VisionError
            raise VisionError(
                message="Vision analysis returned unsuccessful without quality info",
                user_action="Image analysis failed. Please try with a different image."
            )

    # Extract detections and spatial insights - NO defaults
    if "spatial_insights" not in vision_data:
        from src.utils.exceptions import VisionError
        raise VisionError(
            message="Vision agent did not provide spatial insights",
            user_action="Image analysis incomplete. Please try again."
        )

    state.detections = vision_data.get("detections", [])
    state.spatial_insights = vision_data["spatial_insights"]
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


def rag_node(state: AgentState) -> Dict[str, Any]:
    """RAG node: Retrieve evidence and validate response."""
    logger.info("Orchestrator: RAG node")

    result = rag_agent.execute(state=state)

    if result.status.value != "success":
        logger.error(f"Orchestrator: RAG failed - {result.error}")
        # Raise exception - NO fallback
        from src.utils.exceptions import RAGError
        raise RAGError(
            message=f"RAG agent failed: {result.error}",
            user_action="Could not retrieve information from knowledge base. Please try rephrasing your question."
        )

    rag_data = result.data

    # Strict field checking - NO defaults
    if "response" not in rag_data:
        from src.utils.exceptions import RAGError
        raise RAGError(
            message="RAG agent did not provide response",
            user_action="Could not generate response. Please try again."
        )

    state.rag_response = rag_data["response"]
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


def synthesis_node(state: AgentState) -> Dict[str, Any]:
    """Synthesis node: Assemble final response with citations."""
    logger.info("Orchestrator: Synthesis node")

    # Collect all data for synthesis
    synthesis_kwargs = {
        "rag_response": state.rag_response,
        "sources": state.sources,
        "recommendations": state.triage_decision.get("recommendations", []) if state.triage_decision else [],
        "overall_confidence": state.confidence_score,
    }

    result = synthesis_agent.execute(state=state, **synthesis_kwargs)

    if result.status.value != "success":
        logger.error(f"Orchestrator: Synthesis failed - {result.error}")
        # Raise exception - NO fallback
        from src.utils.exceptions import SynthesisError
        raise SynthesisError(
            message=f"Synthesis agent failed: {result.error}",
            user_action="Could not assemble final response. Please try again."
        )

    synthesis_data = result.data

    if "final_response" not in synthesis_data:
        from src.utils.exceptions import SynthesisError
        raise SynthesisError(
            message="Synthesis agent did not provide final response",
            user_action="Could not generate final response. Please try again."
        )

    state.final_response = synthesis_data["final_response"]

    # Add message to history
    state.add_message(MessageRole.ASSISTANT, state.final_response)

    # Save checkpoint
    save_state(state)

    logger.info(f"Orchestrator: Synthesis complete - Response length: {len(state.final_response)}")

    return {
        "final_response": state.final_response,
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

    # Run graph - exceptions will propagate
    result = app.invoke(state)

    # Result is a dict, not an object - strict checking
    if "final_response" not in result:
        logger.error("Orchestrator: No final_response in result")
        raise ValueError("Orchestrator did not produce final_response")

    response = result["final_response"]
    sources = result.get("sources", [])
    confidence = result.get("confidence_score", 0.0)
    conversation_id = result.get("conversation_id", state.conversation_id)

    logger.info(
        f"Orchestrator: Run complete - Response: {len(response)} chars, "
        f"Sources: {len(sources)}, Confidence: {confidence:.2f}"
    )

    return {
        "response": response,
        "sources": [src.model_dump() if hasattr(src, 'model_dump') else src for src in sources],
        "confidence": confidence,
        "conversation_id": conversation_id,
    }