"""RAG Agent with hallucination detection and source validation.

This agent:
- Retrieves relevant evidence from vector store and knowledge graph
- Validates response claims against retrieved context
- Generates confidence scores for evidence-based claims
- Returns structured citations for transparency
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, TYPE_CHECKING

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.agents.specialized.base_agent import BaseAgent
from src.agents.state_models import AgentState, SourceCitation
from src.utils.llm import get_gemini_chat

if TYPE_CHECKING:
    from src.rag import RAGSystem

logger = logging.getLogger(__name__)


class ClaimValidation(BaseModel):
    """Validation result for a single claim."""

    claim: str
    is_supported: bool
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_sources: List[int] = Field(default_factory=list)  # Source IDs


class RAGResult(BaseModel):
    """Structured RAG analysis result."""

    response: str
    sources: List[SourceCitation]
    claim_validations: List[ClaimValidation] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0)
    hallucination_risk: str = Field(description="low, medium, high")
    recommendations: List[str] = Field(default_factory=list)


class RAGAgent(BaseAgent):
    """Agent for evidence-based retrieval using centralized RAG system.

    Note: Validation is handled by RAGSystem.query() to avoid duplication.
    """

    def __init__(self):
        super().__init__(name="RAGAgent")

    def _execute(self, state: AgentState, **kwargs) -> Dict[str, Any]:
        """Execute RAG retrieval and validation."""
        logger.info(f"RAGAgent: Processing query '{state.input[:50]}...'")

        # Step 1: Retrieve documents and generate response
        try:
            # Use cached singleton to avoid reinstantiation overhead
            from src.ui.chat_interface import get_rag_system_singleton

            detections_json = json.dumps([d.model_dump() for d in state.detections]) if state.detections else ""

            # Get cached RAG system (no setup needed, already loaded)
            rag_system = get_rag_system_singleton()
            logger.debug("RAGAgent: Using cached RAGSystem singleton")

            # Build contextualized query for better retrieval
            contextualized_query = self._build_contextualized_query(state)
            logger.debug(f"RAGAgent: Contextualized query: {contextualized_query[:100]}...")

            # Query RAG system
            result = rag_system.query(
                query=contextualized_query,
                detections=detections_json,
                spatial_insights=state.spatial_insights,
                history=[msg.model_dump() for msg in state.history],
                profile=state.user_profile.model_dump(),
            )

            rag_response = result.response
            sources = result.sources

            logger.info(f"RAGAgent: Retrieved {len(sources)} sources")
            logger.debug(f"RAGAgent: Response length: {len(rag_response)} chars")

        except Exception as e:
            logger.error(f"RAGAgent: Retrieval failed - {e}")
            raise

        # Step 2: Use validation from RAGSystem (already computed)
        validation_result = result.validation_result

        if not validation_result:
            logger.error("RAGAgent: No validation result from RAGSystem")
            from src.utils.exceptions import RAGError
            raise RAGError(
                message="RAG system did not provide validation result",
                user_action="The system could not validate the response. Please try again."
            )

        # Strict validation - NO defaults
        if "overall_confidence" not in validation_result:
            logger.error("RAGAgent: Missing overall_confidence in validation result")
            from src.utils.exceptions import RAGError
            raise RAGError(
                message="Validation result missing overall_confidence field",
                user_action="The system could not properly validate the response. Please try again."
            )

        if "hallucination_risk" not in validation_result:
            logger.error("RAGAgent: Missing hallucination_risk in validation result")
            from src.utils.exceptions import RAGError
            raise RAGError(
                message="Validation result missing hallucination_risk field",
                user_action="The system could not properly validate the response. Please try again."
            )

        logger.info(
            f"RAGAgent: Validation from RAGSystem - Confidence={validation_result['overall_confidence']:.2f}, "
            f"Risk={validation_result['hallucination_risk']}"
        )

        # Step 3: Generate recommendations based on confidence
        recommendations = self._generate_recommendations(
            validation_result,
            state.user_profile.symptoms
        )

        return {
            "response": rag_response,
            "sources": sources,
            "claim_validations": validation_result.get("claims", []),
            "overall_confidence": validation_result["overall_confidence"],
            "hallucination_risk": validation_result["hallucination_risk"],
            "recommendations": recommendations,
        }

    def _build_contextualized_query(self, state: AgentState) -> str:
        """Build enhanced query with conversation context for better retrieval.

        Args:
            state: Current agent state

        Returns:
            Contextualized query string
        """
        parts = []

        # Add chief complaint if available
        if state.user_profile.chief_complaint:
            parts.append(f"Chief Complaint: {state.user_profile.chief_complaint}")

        # Add SOCRATES symptoms
        symptoms = state.user_profile.symptoms
        if symptoms.site:
            parts.append(f"Location: {symptoms.site}")
        if symptoms.character:
            parts.append(f"Type: {symptoms.character}")
        if symptoms.severity:
            parts.append(f"Severity: {symptoms.severity}/10")
        if symptoms.onset:
            parts.append(f"Started: {symptoms.onset}")
        if symptoms.time_course:
            parts.append(f"Progression: {symptoms.time_course}")

        # Add current query
        parts.append(f"Query: {state.input}")

        # Combine with newlines for structure
        contextualized = "\n".join(parts)

        return contextualized if len(parts) > 1 else state.input

    def _generate_recommendations(
        self,
        validation_result: Dict[str, Any],
        symptoms
    ) -> List[str]:
        """Generate actionable recommendations based on validation and symptoms."""
        recommendations = []

        # Check hallucination risk (NO defaults - validation_result must have these)
        risk = validation_result["hallucination_risk"]
        confidence = validation_result["overall_confidence"]

        if risk == "high" or confidence < 0.5:
            recommendations.append(
                "⚠️ Low confidence in AI response. Please consult a dentist for accurate diagnosis."
            )

        # Check symptom severity
        severity = getattr(symptoms, 'severity', None)
        if severity and severity >= 7:
            recommendations.append(
                "🚨 High pain severity detected. Seek immediate dental consultation."
            )

        # Check for emergency indicators in symptoms
        associations = getattr(symptoms, 'associations', [])
        if any(keyword in str(associations).lower() for keyword in ['swelling', 'bengkak', 'fever', 'demam']):
            recommendations.append(
                "⚠️ Possible infection signs. Please see a dentist promptly."
            )

        # Always suggest professional consultation
        if not recommendations:  # Only if no urgent ones added
            recommendations.append(
                "💡 For proper diagnosis and treatment, please visit a dentist."
            )

        # Add citation reminder
        if validation_result.get("claims"):
            supported_pct = (
                sum(1 for c in validation_result["claims"] if c.get("is_supported", False))
                / len(validation_result["claims"])
                * 100
            )
            recommendations.append(
                f"📚 {supported_pct:.0f}% of claims supported by sources (see citations below)."
            )

        return recommendations