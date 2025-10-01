"""Advanced claim validation with citation tracing and context grounding.

This module provides:
- Extraction of factual claims from generated responses
- Validation of each claim against source documents
- Citation tracing (link claims to specific source chunks)
- Hallucination risk assessment
- Confidence scoring per claim
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.utils.llm import get_gemini_chat

logger = logging.getLogger(__name__)


class Claim(BaseModel):
    """Individual factual claim extracted from response."""

    claim_text: str = Field(description="Extracted factual statement")
    claim_type: str = Field(
        description="Type: definition, cause, treatment, prevention, symptom"
    )
    is_supported: bool = Field(description="Whether supported by sources")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in support")
    supporting_sources: List[int] = Field(
        default_factory=list,
        description="Source IDs that support this claim"
    )
    supporting_snippets: List[str] = Field(
        default_factory=list,
        description="Exact snippets from sources"
    )
    reasoning: str = Field(default="", description="Why claim is/isn't supported")


class ValidationResult(BaseModel):
    """Complete validation result for a response."""

    claims: List[Claim]
    overall_confidence: float = Field(ge=0.0, le=1.0)
    hallucination_risk: str = Field(description="low, medium, high")
    unsupported_claims: List[str] = Field(default_factory=list)
    well_supported_claims: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class ClaimValidator:
    """Validates generated responses against source documents."""

    CLAIM_EXTRACTION_PROMPT = """Extract factual medical/dental claims from the response.

**Response to Analyze:**
{response}

**Task:**
1. Extract ALL factual claims (ignore greetings, questions, disclaimers)
2. Classify each claim by type
3. Be precise - one claim per statement

**Claim Types:**
- **definition**: What something is (e.g., "Caries is tooth decay")
- **cause**: What causes condition (e.g., "Bacteria cause caries")
- **symptom**: Signs/symptoms (e.g., "Caries causes pain")
- **treatment**: How to treat (e.g., "Fillings treat cavities")
- **prevention**: How to prevent (e.g., "Fluoride prevents caries")

**Output JSON:**
{{
  "claims": [
    {{
      "claim_text": "exact claim statement",
      "claim_type": "definition|cause|symptom|treatment|prevention"
    }}
  ]
}}

Respond with ONLY valid JSON."""

    VALIDATION_PROMPT = """Validate whether a claim is supported by source documents.

**Claim:** {claim_text}

**Source Documents:**
{sources_text}

**Task:**
1. Check if claim is EXPLICITLY stated or STRONGLY IMPLIED in sources
2. Find exact snippets that support the claim
3. Assign confidence:
   - 1.0: Explicitly stated in sources
   - 0.7-0.9: Strongly implied by sources
   - 0.4-0.6: Partially supported or indirectly mentioned
   - 0.0-0.3: Not supported or contradicts sources
4. List source IDs (numbers) that support claim

**Validation Criteria:**
- ✅ Supported: Claim matches source content (exact or paraphrased)
- ⚠️ Partially: Claim generalizes source info (may be too broad)
- ❌ Unsupported: Claim adds new info not in sources OR contradicts sources

**Output JSON:**
{{
  "is_supported": boolean,
  "confidence": 0.0-1.0,
  "supporting_sources": [1, 3],
  "supporting_snippets": ["exact quotes from sources"],
  "reasoning": "brief explanation"
}}

Respond with ONLY valid JSON."""

    def __init__(self):
        """Initialize claim validator."""
        self.llm = get_gemini_chat(
            model="gemini-2.5-flash",
            temperature=0.0,  # Deterministic for validation
        )
        logger.info("ClaimValidator: Initialized")

    def validate(
        self,
        response: str,
        sources: List[Document],
    ) -> ValidationResult:
        """Validate response claims against source documents.

        Args:
            response: Generated response to validate
            sources: Source documents used for generation

        Returns:
            ValidationResult with per-claim validation and overall assessment
        """
        logger.info(f"ClaimValidator: Validating response ({len(response)} chars, {len(sources)} sources)")

        try:
            # Step 1: Extract claims
            claims_data = self._extract_claims(response)

            if not claims_data:
                logger.warning("ClaimValidator: No claims extracted")
                return ValidationResult(
                    claims=[],
                    overall_confidence=0.5,
                    hallucination_risk="medium",
                    recommendations=["No factual claims to validate"],
                )

            logger.info(f"ClaimValidator: Extracted {len(claims_data)} claims")

            # Step 2: Validate each claim
            validated_claims: List[Claim] = []

            for claim_dict in claims_data:
                claim_text = claim_dict.get("claim_text", "")
                claim_type = claim_dict.get("claim_type", "unknown")

                validation = self._validate_claim(claim_text, sources)

                validated_claim = Claim(
                    claim_text=claim_text,
                    claim_type=claim_type,
                    is_supported=validation.get("is_supported", False),
                    confidence=validation.get("confidence", 0.0),
                    supporting_sources=validation.get("supporting_sources", []),
                    supporting_snippets=validation.get("supporting_snippets", []),
                    reasoning=validation.get("reasoning", ""),
                )

                validated_claims.append(validated_claim)

                logger.debug(
                    f"ClaimValidator: Claim '{claim_text[:50]}...' - "
                    f"Supported={validated_claim.is_supported}, "
                    f"Confidence={validated_claim.confidence:.2f}"
                )

            # Step 3: Compute overall metrics
            result = self._compute_validation_metrics(validated_claims)

            logger.info(
                f"ClaimValidator: Overall confidence={result.overall_confidence:.2f}, "
                f"Risk={result.hallucination_risk}, "
                f"Supported={len(result.well_supported_claims)}/{len(validated_claims)}"
            )

            return result

        except Exception as e:
            logger.error(f"ClaimValidator: Validation failed - {e}")
            return ValidationResult(
                claims=[],
                overall_confidence=0.5,
                hallucination_risk="medium",
                recommendations=["Validation error - treat with caution"],
            )

    def _extract_claims(self, response: str) -> List[Dict[str, str]]:
        """Extract factual claims from response.

        Args:
            response: Generated response text

        Returns:
            List of claim dicts with claim_text and claim_type
        """
        try:
            prompt = self.CLAIM_EXTRACTION_PROMPT.format(response=response)

            llm_response = self.llm.invoke(
                [HumanMessage(content=prompt)],
                config={"response_format": {"type": "json_object"}},
            )

            raw = llm_response.content.strip()
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            json_str = json_match.group(0) if json_match else raw

            result = json.loads(json_str)
            claims = result.get("claims", [])

            return claims

        except Exception as e:
            logger.error(f"ClaimValidator: Claim extraction failed - {e}")
            # Fallback: split by periods
            return self._fallback_extract_claims(response)

    def _fallback_extract_claims(self, response: str) -> List[Dict[str, str]]:
        """Fallback claim extraction using simple heuristics.

        Args:
            response: Response text

        Returns:
            List of claim dicts
        """
        # Split by sentence
        sentences = [s.strip() for s in response.split(".") if s.strip()]

        claims = []
        for sent in sentences:
            # Skip greetings, questions, disclaimers
            if any(word in sent.lower() for word in ["halo", "terima kasih", "konsultasi", "dokter gigi", "?"]):
                continue

            # Keep factual-sounding sentences
            if len(sent.split()) >= 5:  # At least 5 words
                claims.append({
                    "claim_text": sent,
                    "claim_type": "unknown"
                })

        return claims[:10]  # Limit to avoid processing too many

    def _validate_claim(
        self,
        claim_text: str,
        sources: List[Document]
    ) -> Dict[str, Any]:
        """Validate a single claim against sources.

        Args:
            claim_text: Claim to validate
            sources: Source documents

        Returns:
            Validation dict with is_supported, confidence, sources, snippets
        """
        try:
            # Format sources for prompt
            sources_text = "\n\n".join([
                f"**Source {i+1}:** {doc.page_content[:300]}..."
                for i, doc in enumerate(sources)
            ])

            prompt = self.VALIDATION_PROMPT.format(
                claim_text=claim_text,
                sources_text=sources_text,
            )

            llm_response = self.llm.invoke(
                [HumanMessage(content=prompt)],
                config={"response_format": {"type": "json_object"}},
            )

            raw = llm_response.content.strip()
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            json_str = json_match.group(0) if json_match else raw

            validation = json.loads(json_str)
            return validation

        except Exception as e:
            logger.error(f"ClaimValidator: Validation error for claim - {e}")
            # Fallback: keyword matching
            return self._fallback_validate_claim(claim_text, sources)

    def _fallback_validate_claim(
        self,
        claim_text: str,
        sources: List[Document]
    ) -> Dict[str, Any]:
        """Fallback validation using keyword matching.

        Args:
            claim_text: Claim to validate
            sources: Source documents

        Returns:
            Basic validation dict
        """
        # Extract keywords from claim (simple approach)
        claim_keywords = set(
            word.lower()
            for word in re.findall(r'\w+', claim_text)
            if len(word) > 3
        )

        # Check keyword overlap with sources
        max_overlap = 0
        supporting_sources = []

        for i, doc in enumerate(sources):
            source_text = doc.page_content.lower()
            overlap = sum(1 for kw in claim_keywords if kw in source_text)

            if overlap > max_overlap:
                max_overlap = overlap

            if overlap >= 2:  # At least 2 keywords match
                supporting_sources.append(i + 1)

        # Compute confidence based on overlap
        if max_overlap >= 4:
            confidence = 0.8
            is_supported = True
        elif max_overlap >= 2:
            confidence = 0.5
            is_supported = True
        else:
            confidence = 0.3
            is_supported = False

        return {
            "is_supported": is_supported,
            "confidence": confidence,
            "supporting_sources": supporting_sources[:3],
            "supporting_snippets": [],
            "reasoning": f"Keyword overlap: {max_overlap} keywords",
        }

    def _compute_validation_metrics(
        self,
        claims: List[Claim]
    ) -> ValidationResult:
        """Compute overall validation metrics.

        Args:
            claims: List of validated claims

        Returns:
            ValidationResult with aggregated metrics
        """
        if not claims:
            return ValidationResult(
                claims=[],
                overall_confidence=0.5,
                hallucination_risk="medium",
            )

        # Compute overall confidence (average)
        overall_confidence = sum(c.confidence for c in claims) / len(claims)

        # Count supported vs unsupported
        supported_count = sum(1 for c in claims if c.is_supported)
        unsupported_pct = (len(claims) - supported_count) / len(claims) * 100

        # Determine hallucination risk
        if unsupported_pct < 20:
            hallucination_risk = "low"
        elif unsupported_pct < 40:
            hallucination_risk = "medium"
        else:
            hallucination_risk = "high"

        # Categorize claims
        well_supported = [
            c.claim_text for c in claims
            if c.is_supported and c.confidence >= 0.7
        ]
        unsupported = [
            c.claim_text for c in claims
            if not c.is_supported or c.confidence < 0.4
        ]

        # Generate recommendations
        recommendations = []
        if hallucination_risk == "high":
            recommendations.append(
                "⚠️ High hallucination risk detected. Verify claims with dentist."
            )
        if unsupported:
            recommendations.append(
                f"❌ {len(unsupported)} unsupported claims found. Use caution."
            )
        if overall_confidence < 0.6:
            recommendations.append(
                "🔴 Low confidence in response accuracy. Consult professional."
            )

        return ValidationResult(
            claims=claims,
            overall_confidence=overall_confidence,
            hallucination_risk=hallucination_risk,
            unsupported_claims=unsupported,
            well_supported_claims=well_supported,
            recommendations=recommendations,
        )