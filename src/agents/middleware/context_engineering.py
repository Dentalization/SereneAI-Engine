"""
Context Engineering Middleware for dynamic prompt and tool selection.
Follows LangChain 1.0 context engineering best practices.
"""

from langchain.agents.middleware import AgentMiddleware, before_model

from src.agents.state import UserProfile


class ContextEngineeringMiddleware(AgentMiddleware):
    """
    Dynamic context injection based on conversation state.

    Implements LangChain 1.0 context engineering patterns:
    - State-driven prompts (adjust based on conversation stage)
    - Dynamic tool selection (enable/disable tools based on context)
    - Model switching (select appropriate model for task)

    Example:
        >>> middleware = ContextEngineeringMiddleware()
        >>> agent = create_agent(..., middleware=[middleware])
    """

    @before_model
    def inject_context(self, state):
        """
        Inject dynamic context before model invocation.

        Adds:
        - Conversation stage information
        - User profile summary
        - Available tools context
        - Emergency detection hints
        """
        # Get user profile
        user_profile: UserProfile = state.get("user_profile")
        if not user_profile:
            return {}

        # Build context summary
        context_parts = []

        # Add symptom context if available
        symptoms = user_profile.symptoms
        if symptoms.completeness_score() > 0:
            context_parts.append(
                f"Patient Symptoms (SOCRATES {symptoms.completeness_score()}/9):"
            )
            if symptoms.site:
                context_parts.append(f"- Location: {symptoms.site}")
            if symptoms.character:
                context_parts.append(f"- Pain type: {symptoms.character}")
            if symptoms.severity is not None:
                context_parts.append(f"- Severity: {symptoms.severity}/10")

        # Add detection context if available
        if user_profile.detections:
            context_parts.append("\nDetected Conditions:")
            for det in user_profile.detections[-3:]:  # Last 3
                context_parts.append(
                    f"- {det.class_name} (confidence: {det.confidence:.2f})"
                )

        # Add medical history if relevant
        if user_profile.medical_conditions:
            context_parts.append(
                f"\nMedical History: {', '.join(user_profile.medical_conditions)}"
            )

        if user_profile.medications:
            context_parts.append(
                f"Current Medications: {', '.join(user_profile.medications)}"
            )

        if user_profile.allergies:
            context_parts.append(f"Allergies: {', '.join(user_profile.allergies)}")

        # Emergency detection hint
        if symptoms.severity and symptoms.severity >= 8:
            context_parts.append(
                "\n⚠️ HIGH SEVERITY: Consider emergency recommendations."
            )

        # Add language preference
        language = user_profile.language
        context_parts.append(
            f"\nLanguage: {'Indonesian (id)' if language == 'id' else 'English (en)'}"
        )

        # Combine context
        if context_parts:
            context_summary = "\n".join(context_parts)

            # Inject as system message (prepend to messages)
            messages = state.get("messages", [])

            # Only inject if not already present
            if not any(
                "SOCRATES" in str(msg) for msg in messages[:2]
            ):  # Check first 2 messages
                from langchain_core.messages import SystemMessage

                context_message = SystemMessage(
                    content=f"=== PATIENT CONTEXT ===\n{context_summary}\n=== END CONTEXT ==="
                )
                messages = [context_message] + messages

                return {"messages": messages}

        return {}
