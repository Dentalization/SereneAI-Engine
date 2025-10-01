"""Synthesis Agent for assembling final responses with citations.

This agent:
- Combines outputs from Vision, RAG, and Anamnesis agents
- Formats responses with proper citations
- Adds empathetic framing and actionable advice
- Ensures multilingual support (Indonesian/English)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from src.agents.specialized.base_agent import BaseAgent
from src.agents.state_models import AgentState, SourceCitation

logger = logging.getLogger(__name__)


class SynthesisResult(BaseModel):
    """Final synthesized response."""

    response: str
    formatted_citations: str
    recommendations: List[str]
    confidence_display: str
    language: str = "id"


class SynthesisAgent(BaseAgent):
    """Agent for final response assembly and formatting."""

    def __init__(self):
        super().__init__(name="SynthesisAgent")

    def _execute(self, state: AgentState, **kwargs) -> Dict[str, Any]:
        """Execute response synthesis."""
        logger.info("SynthesisAgent: Assembling final response")

        # Determine language from profile or input
        language = state.user_profile.language

        # Collect all components
        rag_response = kwargs.get("rag_response", state.rag_response)
        sources = kwargs.get("sources", state.sources)
        recommendations = kwargs.get("recommendations", [])
        confidence = kwargs.get("overall_confidence", state.confidence_score)
        spatial_insights = state.spatial_insights
        detections = state.detections

        # Build response sections
        response_parts = []

        # 1. Image analysis (if present)
        if detections:
            detection_summary = self._format_detections(detections, language)
            response_parts.append(detection_summary)

            if spatial_insights:
                response_parts.append(f"\n**Spatial Analysis:**\n{spatial_insights}")

        # 2. Main RAG response
        if rag_response:
            response_parts.append(f"\n{rag_response}")

        # 3. Recommendations
        if recommendations:
            rec_header = "**Rekomendasi:**" if language == "id" else "**Recommendations:**"
            rec_text = "\n".join([f"- {rec}" for rec in recommendations])
            response_parts.append(f"\n{rec_header}\n{rec_text}")

        # 4. Confidence display
        confidence_display = self._format_confidence(confidence, language)

        # 5. Format citations
        formatted_citations = self._format_citations(sources, language)

        # Combine all parts
        final_response = "\n\n".join(filter(None, response_parts))

        logger.info(f"SynthesisAgent: Final response length: {len(final_response)} chars")
        logger.debug(f"SynthesisAgent: Confidence={confidence:.2f}, Sources={len(sources)}")

        return {
            "final_response": final_response,
            "formatted_citations": formatted_citations,
            "recommendations": recommendations,
            "confidence_display": confidence_display,
            "language": language,
        }

    def _format_detections(self, detections: List, language: str) -> str:
        """Format YOLO detections into readable text."""
        if not detections:
            return ""

        header = "**Hasil Deteksi:**" if language == "id" else "**Detection Results:**"

        detection_lines = []
        for i, det in enumerate(detections, 1):
            class_name = det.class_name if hasattr(det, 'class_name') else det.get('class_name', 'unknown')
            confidence = det.confidence if hasattr(det, 'confidence') else det.get('confidence', 0)

            # Translate class names to Indonesian if needed
            if language == "id":
                class_translation = {
                    "calculus": "karang gigi",
                    "caries": "karies/gigi berlubang",
                    "gingivitis": "radang gusi",
                    "hypodontia": "gigi hilang",
                    "tooth_discoloration": "perubahan warna gigi",
                    "ulcer": "sariawan",
                }
                class_display = class_translation.get(class_name, class_name)
            else:
                class_display = class_name.replace("_", " ").title()

            detection_lines.append(
                f"{i}. {class_display} (confidence: {confidence:.1%})"
            )

        return f"{header}\n" + "\n".join(detection_lines)

    def _format_confidence(self, confidence: float, language: str) -> str:
        """Format confidence score with visual indicator."""
        if confidence >= 0.8:
            if language == "id":
                return "🟢 Tingkat kepercayaan tinggi (>80%)"
            else:
                return "🟢 High confidence (>80%)"
        elif confidence >= 0.6:
            if language == "id":
                return "🟡 Tingkat kepercayaan sedang (60-80%)"
            else:
                return "🟡 Moderate confidence (60-80%)"
        else:
            if language == "id":
                return "🔴 Tingkat kepercayaan rendah (<60%) - Konsultasi dokter gigi disarankan"
            else:
                return "🔴 Low confidence (<60%) - Professional consultation recommended"

    def _format_citations(self, sources: List[SourceCitation], language: str) -> str:
        """Format source citations for display."""
        if not sources:
            return ""

        header = "\n\n---\n📚 **Sumber Referensi:**\n" if language == "id" else "\n\n---\n📚 **References:**\n"

        citation_lines = []
        for src in sources:
            # Determine citation format by provider
            if src.provider == "PubMed":
                citation = f"[{src.id}] {src.title}"
                if src.authors:
                    citation += f" — {src.authors}"
                if src.pmid:
                    citation += f" (PMID: {src.pmid})"
                if src.url:
                    citation += f" [Link]({src.url})"
            else:  # PDF
                citation = f"[{src.id}] {src.title}"
                if src.page:
                    citation += f" (Page {src.page})"
                if src.source_path:
                    citation += f" — {src.source_path}"

            citation_lines.append(citation)

        return header + "\n".join(citation_lines)

    def _add_empathetic_framing(self, response: str, language: str) -> str:
        """Add empathetic framing to response (optional enhancement)."""
        # Could add gentle opening/closing based on context
        # For now, return as-is since agents already handle this
        return response