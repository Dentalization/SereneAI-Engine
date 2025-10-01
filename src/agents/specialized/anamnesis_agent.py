"""Anamnesis Agent for structured symptom extraction using SOCRATES framework.

SOCRATES Framework:
- Site: Location of pain/problem
- Onset: When symptoms started
- Character: Nature of pain (sharp, dull, throbbing, aching)
- Radiation: Does pain spread to other areas
- Associations: Other symptoms (swelling, bleeding, fever)
- Time course: How symptoms changed (constant, intermittent, worsening)
- Exacerbating factors: What makes it worse (hot, cold, pressure)
- Relieving factors: What makes it better (painkillers, rest)
- Severity: Pain scale 1-10
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.agents.specialized.base_agent import BaseAgent
from src.agents.state_models import AgentState, SOCRATESProfile
from src.utils.llm import get_gemini_chat

logger = logging.getLogger(__name__)


class AnamnesisResult(BaseModel):
    """Structured result from anamnesis extraction."""

    socrates: SOCRATESProfile
    missing_elements: List[str] = Field(
        default_factory=list,
        description="SOCRATES elements still needed"
    )
    suggested_question: str = Field(
        description="Next question to ask for missing info"
    )
    completeness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How complete is the anamnesis (0-1)"
    )
    ready_for_diagnosis: bool = Field(
        description="Whether enough info collected for diagnosis"
    )


class AnamnesisAgent(BaseAgent):
    """Agent for extracting structured symptom data using SOCRATES."""

    ANAMNESIS_PROMPT = """You are a dental professional conducting systematic symptom assessment using the SOCRATES framework.

**Conversation History:**
{history_str}

**Current User Input:** {input}

**Existing Profile:**
{current_profile}

**SOCRATES Framework Elements:**
1. **Site (S)**: Where is the dental problem located?
   - Specific tooth? Upper/lower? Left/right?
   - Example: "upper right second molar", "lower front teeth"

2. **Onset (O)**: When did symptoms start?
   - How long ago? Sudden or gradual?
   - Example: "3 days ago", "started gradually last week"

3. **Character (C)**: What does the pain/problem feel like?
   - Sharp, dull, throbbing, aching, burning?
   - Example: "sharp stabbing pain", "constant dull ache"

4. **Radiation (R)**: Does pain spread elsewhere?
   - To jaw, ear, head, neck?
   - Example: "radiates to right ear", "spreads to jaw"

5. **Associations (A)**: Any other symptoms?
   - Swelling, bleeding, sensitivity, fever, bad taste?
   - Example: ["swelling in gum", "bleeding when brushing"]

6. **Time course (T)**: How have symptoms changed?
   - Constant, intermittent, getting worse/better?
   - Example: "comes and goes", "getting worse each day"

7. **Exacerbating factors (E)**: What makes it worse?
   - Hot/cold drinks, chewing, sweet foods, lying down?
   - Example: ["cold drinks", "chewing on that side"]

8. **Relieving factors (R)**: What makes it better?
   - Painkillers, warm compress, avoiding certain foods?
   - Example: ["paracetamol", "avoiding cold drinks"]

9. **Severity (S)**: How bad is the pain (1-10)?
   - 1-3: Mild, 4-6: Moderate, 7-10: Severe
   - Example: 7/10

**Your Task:**
1. Extract all available SOCRATES elements from conversation
2. Identify missing elements
3. Suggest ONE specific follow-up question for highest priority missing element
4. Calculate completeness (count of filled elements / 9)
5. Determine if ready for diagnosis (needs ≥5 elements OR severity ≥7)

**Output JSON Schema:**
{{
  "socrates": {{
    "site": "string or null",
    "onset": "string or null",
    "character": "string or null",
    "radiation": "string or null",
    "associations": ["string"],
    "time_course": "string or null",
    "exacerbating_factors": ["string"],
    "relieving_factors": ["string"],
    "severity": int or null (1-10)
  }},
  "missing_elements": ["element names"],
  "suggested_question": "Empathetic Indonesian question for next missing element",
  "completeness_score": 0.0-1.0,
  "ready_for_diagnosis": boolean
}}

**Guidelines:**
- Be thorough but extract only explicitly stated info (don't infer)
- Use natural Indonesian for suggested questions
- Prioritize: Site > Onset > Severity > Character > Time course
- If severity ≥7 or mentions "sangat sakit"/"emergency", mark ready_for_diagnosis=true

Respond with ONLY valid JSON."""

    def __init__(self):
        super().__init__(name="AnamnesisAgent")
        self.llm = get_gemini_chat(
            model="gemini-2.5-flash",
            temperature=0.1,
        )

    def _execute(self, state: AgentState, **kwargs) -> Dict[str, Any]:
        """Execute anamnesis extraction."""
        logger.info(f"AnamnesisAgent: Extracting symptoms from input")

        # Prepare context
        history_str = state.get_history_string(last_n=10)  # More history for anamnesis
        current_profile = state.user_profile.model_dump()

        prompt = self.ANAMNESIS_PROMPT.format(
            history_str=history_str,
            input=state.input,
            current_profile=json.dumps(current_profile, indent=2),
        )

        try:
            response = self.llm.invoke(
                [HumanMessage(content=prompt)],
                config={"response_format": {"type": "json_object"}},
            )
            raw_response = response.content.strip()
            logger.debug(f"AnamnesisAgent: Raw response: {raw_response}")

            # Parse JSON
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            json_str = json_match.group(0) if json_match else raw_response
            result_dict = json.loads(json_str)

            # Validate and structure
            anamnesis_result = AnamnesisResult(**result_dict)

            logger.info(
                f"AnamnesisAgent: Extracted - "
                f"Completeness={anamnesis_result.completeness_score:.2f}, "
                f"Ready={anamnesis_result.ready_for_diagnosis}, "
                f"Missing={len(anamnesis_result.missing_elements)} elements"
            )

            # Log extracted elements
            socrates_dict = anamnesis_result.socrates.model_dump()
            filled = [k for k, v in socrates_dict.items() if v]
            logger.debug(f"AnamnesisAgent: Filled SOCRATES elements: {filled}")
            logger.debug(f"AnamnesisAgent: Missing elements: {anamnesis_result.missing_elements}")

            return {
                "anamnesis_result": anamnesis_result,
                "socrates": anamnesis_result.socrates,
                "suggested_question": anamnesis_result.suggested_question,
                "completeness_score": anamnesis_result.completeness_score,
                "ready_for_diagnosis": anamnesis_result.ready_for_diagnosis,
                "profile_update": {
                    "symptoms": anamnesis_result.socrates.model_dump()
                },
            }

        except json.JSONDecodeError as e:
            logger.error(f"AnamnesisAgent: JSON parse error - {e}")
            logger.debug(f"AnamnesisAgent: Problematic response: {raw_response}")
            return self._fallback_extraction(state)

        except Exception as e:
            logger.error(f"AnamnesisAgent: Execution error - {e}")
            raise

    def _fallback_extraction(self, state: AgentState) -> Dict[str, Any]:
        """Fallback extraction using simple heuristics."""
        logger.warning("AnamnesisAgent: Using fallback extraction")

        # Simple keyword-based extraction
        input_lower = state.input.lower()

        # Extract severity from numbers
        severity = None
        severity_match = re.search(r'(\d+)/10|skala (\d+)|severity (\d+)', input_lower)
        if severity_match:
            severity = int([g for g in severity_match.groups() if g][0])

        # Keywords for character
        character = None
        if any(word in input_lower for word in ['tajam', 'sharp', 'menusuk']):
            character = "sharp pain"
        elif any(word in input_lower for word in ['berdenyut', 'throb', 'pulsing']):
            character = "throbbing pain"
        elif any(word in input_lower for word in ['tumpul', 'dull', 'aching']):
            character = "dull ache"

        # Basic associations
        associations = []
        if any(word in input_lower for word in ['bengkak', 'swell']):
            associations.append("swelling")
        if any(word in input_lower for word in ['berdarah', 'bleed']):
            associations.append("bleeding")
        if any(word in input_lower for word in ['demam', 'fever']):
            associations.append("fever")

        socrates = SOCRATESProfile(
            character=character,
            severity=severity,
            associations=associations,
        )

        # Count filled elements
        filled_count = sum(
            1 for v in socrates.model_dump().values()
            if v and (not isinstance(v, list) or len(v) > 0)
        )
        completeness = filled_count / 9.0

        ready = completeness >= 0.5 or (severity is not None and severity >= 7)

        return {
            "anamnesis_result": None,
            "socrates": socrates,
            "suggested_question": "Bisa ceritakan di mana lokasi sakit giginya?",
            "completeness_score": completeness,
            "ready_for_diagnosis": ready,
            "profile_update": {
                "symptoms": socrates.model_dump()
            },
        }