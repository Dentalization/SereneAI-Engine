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
    """Agent for evidence-based retrieval with hallucination detection."""

    VALIDATION_PROMPT = """You are a fact-checker verifying dental advice against source documents.

**Generated Response:**
{response}

**Retrieved Source Documents:**
{sources_text}

**Your Task:**
1. Extract individual factual claims from the response
2. For each claim, check if it's supported by source documents
3. Assign confidence: 1.0 (explicitly stated), 0.7 (strongly implied), 0.5 (weakly implied), 0.0 (unsupported)
4. List source IDs that support each claim

**Output JSON Schema:**
{{
  "claims": [
    {{
      "claim": "extracted factual statement",
      "is_supported": boolean,
      "confidence": 0.0-1.0,
      "supporting_sources": [1, 2]
    }}
  ],
  "overall_confidence": 0.0-1.0,
  "hallucination_risk": "low|medium|high",
  "issues": ["any unsupported or questionable claims"]
}}

**Guidelines:**
- Only validate factual medical/dental claims (not greetings/questions)
- Mark as unsupported if claim contradicts sources or adds new info not in sources
- Overall confidence = average of individual claims
- Hallucination risk: low (<20% unsupported), medium (20-40%), high (>40%)

Respond with ONLY valid JSON."""

    def __init__(self):
        super().__init__(
            name="RAGAgent",
            max_retries=2,
            retry_delay=1.0,
            enable_circuit_breaker=True,
        )
        self.llm = get_gemini_chat(
            model="gemini-2.5-flash",
            temperature=0.0,  # Deterministic for validation
        )

    def _execute(self, state: AgentState, **kwargs) -> Dict[str, Any]:
        """Execute RAG retrieval and validation."""
        logger.info(f"RAGAgent: Processing query '{state.input[:50]}...'")

        # Step 1: Retrieve documents and generate response
        try:
            # Lazy import to avoid circular dependency
            from src.rag import RAGSystem

            detections_json = json.dumps([d.model_dump() for d in state.detections]) if state.detections else ""

            # Initialize RAG system
            rag_system = RAGSystem()
            rag_system.setup()

            # Query RAG system
            result = rag_system.query(
                query=state.input,
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

        # Step 2: Validate response against sources
        try:
            validation_result = self._validate_claims(rag_response, sources)

            logger.info(
                f"RAGAgent: Validation - Confidence={validation_result['overall_confidence']:.2f}, "
                f"Risk={validation_result['hallucination_risk']}, "
                f"Claims={len(validation_result.get('claims', []))}"
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

        except Exception as e:
            logger.warning(f"RAGAgent: Validation failed, continuing without - {e}")
            # Return without validation if it fails
            return {
                "response": rag_response,
                "sources": sources,
                "claim_validations": [],
                "overall_confidence": 0.7,  # Default moderate confidence
                "hallucination_risk": "medium",
                "recommendations": ["Please consult a dentist for professional diagnosis"],
            }

    def _validate_claims(self, response: str, sources: List[SourceCitation]) -> Dict[str, Any]:
        """Validate response claims against source documents."""
        logger.info("RAGAgent: Validating claims against sources")

        # Prepare sources text
        sources_text = "\n\n".join([
            f"**Source {src.id} ({src.provider}):** {src.snippet}"
            for src in sources
        ])

        prompt = self.VALIDATION_PROMPT.format(
            response=response,
            sources_text=sources_text,
        )

        try:
            llm_response = self.llm.invoke(
                [HumanMessage(content=prompt)],
                config={"response_format": {"type": "json_object"}},
            )
            raw = llm_response.content.strip()

            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            json_str = json_match.group(0) if json_match else raw

            validation = json.loads(json_str)
            logger.debug(f"RAGAgent: Validation result: {validation}")

            return validation

        except Exception as e:
            logger.error(f"RAGAgent: Claim validation LLM call failed - {e}")
            # Fallback: assume moderate confidence
            return {
                "claims": [],
                "overall_confidence": 0.7,
                "hallucination_risk": "medium",
                "issues": ["Validation unavailable"],
            }

    def _generate_recommendations(
        self,
        validation_result: Dict[str, Any],
        symptoms
    ) -> List[str]:
        """Generate actionable recommendations based on validation and symptoms."""
        recommendations = []

        # Check hallucination risk
        risk = validation_result.get("hallucination_risk", "medium")
        confidence = validation_result.get("overall_confidence", 0.7)

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