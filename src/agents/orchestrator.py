"""Agent orchestration using LangGraph.

This module defines the conversation state and the graph nodes that coordinate
between prompt orchestration, image analysis, and RAG summarization. Behavior
remains identical to the original implementation; changes focus on readability
and maintainability (docstrings, typing, minor structure).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, TypedDict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from src.config import load_config
from src.tools.rag_tool import query_rag
from src.tools.yolo_tool import detect_issues
from src.utils.llm import get_gemini_chat

logger = logging.getLogger(__name__)

config = load_config()

# Initialize LangChain ChatGoogleGenerativeAI model via utility
model = get_gemini_chat(
    model="gemini-2.5-flash",
    temperature=0.1,
    convert_system_message_to_human=True,
)

# JSON output parser for structured responses
json_parser = JsonOutputParser()


class AgentState(TypedDict):
    """State passed between graph nodes during a single run."""

    input: str
    image_path: str
    detections: str
    spatial_insights: str
    rag_response: str
    final_response: str
    sources: List[Dict[str, Any]]
    history: List[dict]
    conversation_stage: str
    user_profile: Dict[str, Any]


ORCHESTRATOR_PROMPT = """
You are the coordinator for a dental chatbot in Indonesia. Analyze full history: {history_str}
Current input: {input}. Image? {image_flag}. Profile: {profile}.

Reason step-by-step internally:
1. Stage: 'greeting' for first/casual; 'anamnesis' for symptoms collection; 'diagnosis' if 3+ dental details in profile.
2. Update profile: Add/extract from input (e.g., symptoms: sakit gigi).
3. Action: 'greet' = ramah + chief question; 'question' = 1 follow-up (durasi/RPD); 'rag' = advice; 'yolo' = image; 'end' = wrap-up.
4. Next must be single string: 'end' for greet/question; 'validator' for yolo; 'summarizer' for rag.

Example for question: {{"stage": "anamnesis", "action": "question", "response": "Sudah berapa lama?", "next": "end", "params": {{"profile_update": {{"duration": "2 minggu"}}}}}}

OUTPUT ONLY VALID JSON (no extra text): {{"stage": "str", "action": "str", "response": "str or null", "next": "str", "params": {{"profile_update": {{}}}}}}.

Natural, empathetic, suggest consult.
"""


def image_validator(state: AgentState) -> AgentState:
    """Validate image input if present; otherwise pass state through."""
    logger.info("FLOW: Entering image_validator")
    if state.get("image_path"):
        try:
            from src.tools.yolo_tool import validate_image

            validate_image(state["image_path"])
            logger.info(
                "FLOW: Image validated successfully for path: %s", state["image_path"]
            )
            return {"detections": "Valid, proceeding.", "conversation_stage": "diagnosis"}  # type: ignore[return-value]
        except Exception as e:  # noqa: BLE001 - bubble as final_response
            logger.error("FLOW: Image validation failed: %s", str(e))
            return {"final_response": f"Image error: {str(e)}"}  # type: ignore[return-value]
    else:
        logger.debug("FLOW: No image path, skipping validator")
    return state


def yolo_detector(state: AgentState) -> AgentState:
    """Run YOLO detection and spatial analysis if an image is provided."""
    logger.info("FLOW: Entering yolo_detector")
    if state.get("image_path"):
        try:
            detections, _, spatial_insights = detect_issues(state["image_path"])
            state["detections"] = detections
            state["spatial_insights"] = spatial_insights
            state["conversation_stage"] = "diagnosis"
            logger.info(
                "FLOW: YOLO detections: %s... Spatial: %s...",
                detections[:100],
                spatial_insights[:100],
            )
            return state
        except Exception as e:  # noqa: BLE001 - handle gracefully
            logger.error("FLOW: YOLO detection failed: %s", str(e))
            return {"final_response": "Detection error."}  # type: ignore[return-value]
    else:
        logger.debug("FLOW: No image path, skipping detector")
    return state


def rag_summarizer(state: AgentState) -> Dict[str, Any]:
    """Generate a RAG summary and return final response + sources."""
    logger.info("FLOW: Entering rag_summarizer")
    query = state["input"]
    detections = state.get("detections", "")
    spatial_insights = state.get("spatial_insights", "")
    history = state.get("history", [])
    profile = state.get("user_profile", {})
    logger.debug(
        "FLOW: RAG input - Query: %s, Profile: %s...",
        query,
        json.dumps(profile)[:200],
    )
    try:
        response, sources = query_rag(query, detections, spatial_insights, history, profile)
        state["sources"] = sources  # Store for UI
        logger.info(
            "FLOW: RAG response generated (length: %d chars, sources: %d)",
            len(response),
            len(sources),
        )
        return {
            "final_response": response,
            "conversation_stage": "diagnosis",
            "sources": sources,
        }
    except Exception as e:  # noqa: BLE001 - safe fallback
        logger.error("FLOW: RAG summarizer failed: %s", str(e))
        return {
            "final_response": "Saya sarankan periksa dokter gigi segera untuk keluhan ini.",
            "sources": [],
        }


def orchestrator(state: AgentState) -> Dict[str, Any]:
    """Main policy node that routes to validation, detection, or RAG."""
    logger.info(
        "FLOW: Entering orchestrator - Input: %s, Stage: %s",
        state["input"],
        state.get("conversation_stage", "unknown"),
    )
    history = state.get("history", [])
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]])
    image_flag = "Yes" if state.get("image_path") else "No"
    profile = state.get("user_profile", {})

    # Create prompt using LangChain ChatPromptTemplate
    chat_prompt = ChatPromptTemplate.from_messages([("human", ORCHESTRATOR_PROMPT)])

    # Format the prompt with variables
    formatted_prompt = chat_prompt.format_messages(
        history_str=history_str,
        input=state["input"],
        image_flag=image_flag,
        profile=json.dumps(profile),
    )

    logger.debug("GEMINI: Prompt sent (length: %d chars)", len(str(formatted_prompt)))
    try:
        # Invoke model with formatted prompt and request JSON output
        response = model.invoke(
            formatted_prompt,
            config={"response_format": {"type": "json_object"}},
        )
        raw_response = response.content.strip()
        logger.info("GEMINI: Raw response: %s", raw_response)

        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        json_str = json_match.group(0) if json_match else raw_response

        decision = json.loads(json_str)
        logger.info("GEMINI: Parsed decision: %s", decision)

        state["conversation_stage"] = decision.get("stage", "anamnesis")
        profile_update = decision.get("params", {}).get("profile_update", {})
        state["user_profile"] = {**profile, **profile_update}
        logger.info("FLOW: Profile updated: %s", json.dumps(profile_update))

        if decision["action"] in ["greet", "question"]:
            state["final_response"] = decision.get(
                "response", "Ceritakan keluhan gigi Anda?"
            )
            state["sources"] = []  # No sources for non-RAG
            logger.info(
                "FLOW: Action %s - Response: %s...",
                decision["action"],
                state["final_response"][:100],
            )
            return {"next": "end", "final_response": state["final_response"], "sources": []}
        elif decision["action"] == "yolo":
            logger.info("FLOW: Routing to yolo/validator")
            return {"next": "validator"}
        elif decision["action"] == "rag":
            logger.info("FLOW: Routing to rag/summarizer")
            return {"next": "summarizer"}
        else:
            logger.info("FLOW: Default end action")
            return {
                "next": "end",
                "final_response": decision.get(
                    "response", "Terima kasih! Konsultasikan ke dokter gigi ya."
                ),
                "sources": [],
            }

    except json.JSONDecodeError as je:
        logger.error("GEMINI: JSON parse error: %s. Raw: %s", je, raw_response)
        return {
            "next": "end",
            "final_response": "Maaf, saya perlu klarifikasi. Apa keluhan gigi utama Anda?",
            "sources": [],
        }
    except Exception as e:  # noqa: BLE001 - robust fallback
        logger.error("FLOW: Orchestrator error: %s", str(e))
        return {
            "next": "end",
            "final_response": "Error. Coba lagi dengan cerita keluhan gigi Anda.",
            "sources": [],
        }


# Graph definition
graph = StateGraph(AgentState)
graph.add_node("validator", image_validator)
graph.add_node("detector", yolo_detector)
graph.add_node("summarizer", rag_summarizer)
graph.add_node("orchestrator", orchestrator)

graph.set_entry_point("orchestrator")
graph.add_conditional_edges(
    "orchestrator",
    lambda s: s.get("next", "end"),
    {
        "validator": "validator",
        "detector": "detector",
        "summarizer": "summarizer",
        "end": END,
    },
)
graph.add_edge("validator", "detector")
graph.add_edge("detector", "summarizer")
graph.add_edge("summarizer", END)

app = graph.compile()


def run_agent(input_text: str, image_path: str | None = None, history: List[dict] | None = None) -> Dict[str, Any]:
    """Public entrypoint used by the UI to run a single agent turn.

    Args:
        input_text: User input text.
        image_path: Optional path to an image for analysis.
        history: Prior messages for context.

    Returns:
        Dict with keys 'response' and 'sources'.
    """
    logger.info(
        "AGENT: Starting run - Input: %s, Image: %s, History len: %d",
        input_text,
        image_path,
        len(history or []),
    )
    state: AgentState = {
        "input": input_text,
        "image_path": image_path or "",
        "history": history or [],
        "conversation_stage": "greeting",
        "user_profile": {},
        "sources": [],  # Initial
    }
    try:
        result = app.invoke(state)
        response = result.get("final_response", "Error occurred.")
        sources = result.get("sources", [])
        logger.info(
            "AGENT: Run complete - Response length: %d, Sources: %d (breakdown: PDF %d, PubMed %d)",
            len(response),
            len(sources),
            len([s for s in sources if "PDF" in s.get("source", "")]),
            len([s for s in sources if "PubMed" in s.get("source", "")]),
        )
        return {"response": response, "sources": sources}
    except Exception as e:  # noqa: BLE001 - user-facing fallback
        logger.error("AGENT: Run failed: %s", str(e))
        return {
            "response": "Maaf, ada kesalahan teknis. Coba ulangi pertanyaan Anda.",
            "sources": [],
        }
