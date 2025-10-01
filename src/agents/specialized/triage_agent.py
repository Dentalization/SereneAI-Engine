"""Triage Agent for query classification and routing decisions.

This agent analyzes user input to determine:
- Conversation stage (greeting, anamnesis, diagnosis, referral)
- Required action (greet, question, yolo, rag, end)
- Confidence score for routing decision
- Next node in the LangGraph flow
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.agents.specialized.base_agent import BaseAgent
from src.agents.state_models import AgentState, ConversationStage
from src.utils.llm import get_gemini_chat

logger = logging.getLogger(__name__)


class TriageDecision(BaseModel):
    """Structured triage decision output."""

    stage: ConversationStage = Field(description="Current conversation stage")
    action: str = Field(description="Action to take: greet, question, yolo, rag, end")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in decision")
    reasoning: str = Field(description="Explanation of decision")
    next_node: str = Field(description="Next LangGraph node: validator, detector, summarizer, end")
    response: str | None = Field(None, description="Direct response for greet/question actions")
    profile_update: Dict[str, Any] = Field(default_factory=dict, description="User profile updates")


class TriageAgent(BaseAgent):
    """Agent for classifying queries and making routing decisions."""

    TRIAGE_PROMPT = """You are a triage coordinator for a dental chatbot in Indonesia.

Analyze the conversation context and current user input to make a routing decision.

**Conversation History (last 5 messages):**
{history_str}

**Current User Input:** {input}

**Image Attached:** {image_flag}

**Current User Profile:**
{profile}

**Decision Framework:**
1. **Stage Classification:**
   - greeting: First interaction or casual greeting
   - anamnesis: Collecting symptoms (need 3+ SOCRATES elements)
   - diagnosis: Sufficient info for analysis (3+ symptoms OR image present)
   - referral: Emergency or complex case requiring dentist

2. **Action Selection:**
   - greet: Warm welcome + ask chief complaint
   - question: Ask follow-up (use SOCRATES: site, onset, character, duration, severity, etc.)
   - yolo: Process image (if present and not yet processed)
   - rag: Retrieve evidence and provide advice
   - end: Close interaction

3. **Confidence Scoring:**
   - 0.9-1.0: Clear indicators, complete info
   - 0.7-0.8: Reasonable inference, some ambiguity
   - 0.5-0.6: Uncertain, default to conservative action
   - <0.5: Fallback to asking clarifying question

4. **Next Node Routing:**
   - validator -> detector -> summarizer (for images)
   - summarizer (for RAG)
   - end (for greet/question)

**Output JSON Schema:**
{{
  "stage": "greeting|anamnesis|diagnosis|referral",
  "action": "greet|question|yolo|rag|end",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of decision",
  "next_node": "validator|summarizer|end",
  "response": "Direct response text (only for greet/question actions)",
  "profile_update": {{
    "chief_complaint": "extracted complaint",
    "symptoms": {{
      "site": "location",
      "onset": "when started",
      "severity": 1-10
    }}
  }}
}}

**Guidelines:**
- Be empathetic and natural (Indonesian context)
- Extract SOCRATES elements systematically
- Suggest dentist visit for emergencies (severe pain, swelling, bleeding, trauma)
- Use RAG only when sufficient info for meaningful advice
- Update profile incrementally from user responses

Respond with ONLY valid JSON, no additional text."""

    def __init__(self):
        super().__init__(name="TriageAgent")
        self.llm = get_gemini_chat(
            model="gemini-2.5-flash",
            temperature=0.1,
        )
        self.parser = JsonOutputParser(pydantic_object=TriageDecision)

    def _execute(self, state: AgentState, **kwargs) -> Dict[str, Any]:
        """Execute triage classification."""
        logger.info(f"TriageAgent: Processing input '{state.input[:50]}...'")

        # Prepare prompt variables
        history_str = state.get_history_string(last_n=5)
        image_flag = "Yes" if state.image_path else "No"
        profile_dict = state.user_profile.model_dump(mode='json')

        prompt = self.TRIAGE_PROMPT.format(
            history_str=history_str,
            input=state.input,
            image_flag=image_flag,
            profile=json.dumps(profile_dict, indent=2),
        )

        # Invoke LLM with JSON mode
        try:
            response = self.llm.invoke(
                [HumanMessage(content=prompt)],
                config={"response_format": {"type": "json_object"}},
            )
            raw_response = response.content.strip()
            logger.debug(f"TriageAgent: Raw LLM response: {raw_response}")

            # Extract JSON (handle potential wrapper text)
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            json_str = json_match.group(0) if json_match else raw_response

            decision_dict = json.loads(json_str)
            decision = TriageDecision(**decision_dict)

            logger.info(
                f"TriageAgent: Decision - stage={decision.stage}, action={decision.action}, "
                f"confidence={decision.confidence:.2f}, next={decision.next_node}"
            )
            logger.debug(f"TriageAgent: Reasoning - {decision.reasoning}")

            return {
                "decision": decision,
                "stage": decision.stage,
                "action": decision.action,
                "confidence": decision.confidence,
                "next_node": decision.next_node,
                "response": decision.response,
                "profile_update": decision.profile_update,
            }

        except json.JSONDecodeError as e:
            logger.error(f"TriageAgent: JSON parse error - {e}")
            logger.debug(f"TriageAgent: Problematic response: {raw_response}")
            # Raise clear exception - NO fallback to rules
            from src.utils.exceptions import TriageError
            raise TriageError(
                message=f"Failed to parse triage decision from LLM: {str(e)}",
                user_action="Please try rephrasing your question or providing more details about your dental concern."
            ) from e

        except Exception as e:
            logger.error(f"TriageAgent: Execution error - {e}")
            raise