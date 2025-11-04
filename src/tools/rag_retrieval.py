"""
RAG retrieval tool using @tool decorator with ToolRuntime.
Evidence-based dental information retrieval with claim validation.
"""

from typing import Any

from langchain_core.tools import ToolRuntime, tool

from src.agents.state import SourceCitation, UserProfile
from src.config import get_settings
from src.rag.system import RAGSystem


def _build_contextualized_query(query: str, user_profile: UserProfile) -> str:
    """Build query enriched with user context."""
    context_parts = [query]

    # Add symptom context
    symptoms = user_profile.symptoms
    if symptoms.site:
        context_parts.append(f"Location: {symptoms.site}")
    if symptoms.character:
        context_parts.append(f"Pain type: {symptoms.character}")
    if symptoms.severity:
        context_parts.append(f"Severity: {symptoms.severity}/10")

    # Add detection context
    if user_profile.detections:
        detected_conditions = [d.class_name for d in user_profile.detections]
        context_parts.append(f"Detected conditions: {', '.join(detected_conditions)}")

    return " | ".join(context_parts)


@tool
def rag_retrieval(
    query: str,
    *,
    runtime: ToolRuntime,
) -> dict[str, Any]:
    """
    Retrieve evidence-based dental information with claim validation.

    This tool searches the knowledge base (dental guidelines, PubMed articles)
    for relevant information and validates claims against sources to prevent
    hallucinations.

    Args:
        query: Question or topic to search for
        runtime: Injected runtime context (state, context, store, stream)

    Returns:
        Dictionary containing:
        - response: Generated answer based on retrieved evidence
        - sources: List of source citations with provenance
        - validation: Claim validation results (support confidence, hallucination risk)
        - confidence: Overall response confidence (0-1)
        - recommendations: Clinical recommendations based on evidence

    Example:
        >>> result = rag_retrieval("What causes tooth sensitivity?")
        >>> print(result["response"])
        "Tooth sensitivity is caused by..."
        >>> print(result["sources"][0]["title"])
        "Dental Hypersensitivity Guidelines"
    """
    config = get_settings()

    # Stream progress
    runtime.stream_writer(
        {
            "type": "progress",
            "stage": "rag",
            "message": "Mencari informasi medis..." if runtime.context.get("language") == "id" else "Searching medical information...",
        }
    )

    # Access conversation state
    state = runtime.state
    user_profile: UserProfile = state.get("user_profile", UserProfile())
    language = runtime.context.get("language", "id")

    # Build contextualized query
    full_query = _build_contextualized_query(query, user_profile)

    runtime.stream_writer(
        {
            "type": "progress",
            "stage": "rag",
            "message": "Menganalisis sumber...",
        }
    )

    # Initialize RAG system (singleton, cached)
    rag_system = RAGSystem.get_instance()

    # Retrieve documents
    top_k = runtime.context.get("rag_top_k", config.rag_top_k)
    similarity_threshold = runtime.context.get(
        "rag_similarity_threshold", config.rag_similarity_threshold
    )

    retrieved_docs = rag_system.retrieve(
        query=full_query,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )

    if not retrieved_docs:
        return {
            "response": (
                "Maaf, saya tidak menemukan informasi yang relevan di database."
                if language == "id"
                else "Sorry, I couldn't find relevant information in the database."
            ),
            "sources": [],
            "validation": {"support_confidence": 0.0, "hallucination_risk": "high"},
            "confidence": 0.0,
            "recommendations": [
                "Silakan konsultasikan dengan dokter gigi untuk informasi lebih lanjut."
                if language == "id"
                else "Please consult with a dentist for more information."
            ],
        }

    runtime.stream_writer(
        {
            "type": "progress",
            "stage": "rag",
            "message": "Memvalidasi klaim...",
        }
    )

    # Validate claims (hallucination detection)
    validation_result = rag_system.validate_claims(query, retrieved_docs)

    # Generate response with citations
    response = rag_system.generate_response(
        query=query,
        documents=retrieved_docs,
        language=language,
    )

    # Build source citations
    sources = [
        SourceCitation(
            title=doc.metadata.get("title", "Unknown"),
            provider=doc.metadata.get("source", "Unknown"),
            snippet=doc.page_content[:200] + "...",
            confidence=doc.metadata.get("score", 0.0),
            page_number=doc.metadata.get("page"),
            url=doc.metadata.get("url"),
            authors=doc.metadata.get("authors", []),
        )
        for doc in retrieved_docs[:5]  # Top 5 sources
    ]

    # Generate clinical recommendations
    recommendations = _generate_recommendations(
        retrieved_docs, user_profile, language
    )

    # Calculate overall confidence
    confidence = min(
        validation_result.get("support_confidence", 0.0),
        sum(s.confidence for s in sources) / len(sources) if sources else 0.0,
    )

    # Store retrieval in long-term memory (for personalization)
    runtime.store.put(
        namespace=("user", user_profile.user_id, "rag_history"),
        key=query,
        value={
            "query": query,
            "sources": [s.model_dump() for s in sources],
            "confidence": confidence,
            "timestamp": runtime.config.get("timestamp"),
        },
    )

    runtime.stream_writer(
        {
            "type": "progress",
            "stage": "rag",
            "message": "Selesai!",
        }
    )

    return {
        "response": response,
        "sources": [s.model_dump() for s in sources],
        "validation": validation_result,
        "confidence": confidence,
        "recommendations": recommendations,
    }


def _generate_recommendations(
    documents: list,
    user_profile: UserProfile,
    language: str,
) -> list[str]:
    """Generate clinical recommendations based on retrieved evidence."""
    recommendations = []

    # Check for emergency keywords in documents
    emergency_keywords = ["emergency", "urgent", "immediate", "darurat", "segera"]
    has_emergency = any(
        keyword in doc.page_content.lower()
        for doc in documents
        for keyword in emergency_keywords
    )

    if has_emergency or (user_profile.symptoms.severity or 0) >= 8:
        recommendations.append(
            "⚠️ PENTING: Kondisi Anda mungkin memerlukan perhatian segera. Hubungi dokter gigi sesegera mungkin."
            if language == "id"
            else "⚠️ IMPORTANT: Your condition may require immediate attention. Contact a dentist as soon as possible."
        )

    # Add evidence-based recommendations
    if user_profile.symptoms.associations:
        if "swelling" in user_profile.symptoms.associations or "pembengkakan" in user_profile.symptoms.associations:
            recommendations.append(
                "Kompres dingin dapat membantu mengurangi pembengkakan."
                if language == "id"
                else "Cold compress may help reduce swelling."
            )

    if user_profile.symptoms.severity and user_profile.symptoms.severity >= 5:
        recommendations.append(
            "Pertimbangkan obat pereda nyeri yang dijual bebas (konsultasikan dosis dengan apoteker)."
            if language == "id"
            else "Consider over-the-counter pain relief (consult pharmacist for dosage)."
        )

    # Always add general recommendation
    recommendations.append(
        "Jaga kebersihan mulut dengan menyikat gigi 2x sehari dan flossing."
        if language == "id"
        else "Maintain oral hygiene by brushing twice daily and flossing."
    )

    return recommendations
