# orchestrator.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
import re
from src.tools.yolo_tool import detect_issues
from src.tools.rag_tool import query_rag
from src.config import load_config
import logging

config = load_config()

# Initialize LangChain ChatGoogleGenerativeAI model
model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key=config['gemini_api_key'],
    temperature=0.1,
    convert_system_message_to_human=True
)

# JSON output parser for structured responses
json_parser = JsonOutputParser()


class AgentState(TypedDict):
    input: str
    image_path: str
    detections: str
    spatial_insights: str
    rag_response: str
    final_response: str
    sources: List[Dict]  # New: For RAG sources
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


def image_validator(state: AgentState):
    logging.info("FLOW: Entering image_validator")
    if state.get('image_path'):
        try:
            from src.tools.yolo_tool import validate_image
            validate_image(state['image_path'])
            logging.info(f"FLOW: Image validated successfully for path: {state['image_path']}")
            return {"detections": "Valid, proceeding.", "conversation_stage": "diagnosis"}
        except Exception as e:
            logging.error(f"FLOW: Image validation failed: {str(e)}")
            return {"final_response": f"Image error: {str(e)}"}
    else:
        logging.debug("FLOW: No image path, skipping validator")
    return state


def yolo_detector(state: AgentState):
    logging.info("FLOW: Entering yolo_detector")
    if state.get('image_path'):
        try:
            detections, _, spatial_insights = detect_issues(state['image_path'])
            state['detections'] = detections
            state['spatial_insights'] = spatial_insights
            state['conversation_stage'] = "diagnosis"
            logging.info(f"FLOW: YOLO detections: {detections[:100]}... Spatial: {spatial_insights[:100]}...")
            return state
        except Exception as e:
            logging.error(f"FLOW: YOLO detection failed: {str(e)}")
            return {"final_response": "Detection error."}
    else:
        logging.debug("FLOW: No image path, skipping detector")
    return state


def rag_summarizer(state: AgentState):
    logging.info("FLOW: Entering rag_summarizer")
    query = state['input']
    detections = state.get('detections', '')
    spatial_insights = state.get('spatial_insights', '')
    history = state.get('history', [])
    profile = state.get('user_profile', {})
    logging.debug(f"FLOW: RAG input - Query: {query}, Profile: {json.dumps(profile)[:200]}...")
    try:
        response, sources = query_rag(query, detections, spatial_insights, history, profile)
        state['sources'] = sources  # Store for UI
        logging.info(f"FLOW: RAG response generated (length: {len(response)} chars, sources: {len(sources)})")
        return {"final_response": response, "conversation_stage": "diagnosis", "sources": sources}
    except Exception as e:
        logging.error(f"FLOW: RAG summarizer failed: {str(e)}")
        return {"final_response": "Saya sarankan periksa dokter gigi segera untuk keluhan ini.", "sources": []}


def orchestrator(state: AgentState):
    logging.info(
        f"FLOW: Entering orchestrator - Input: {state['input']}, Stage: {state.get('conversation_stage', 'unknown')}")
    history = state.get('history', [])
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]])
    image_flag = "Yes" if state.get('image_path') else "No"
    profile = state.get('user_profile', {})
    
    # Create prompt using LangChain ChatPromptTemplate
    chat_prompt = ChatPromptTemplate.from_messages([
        ("human", ORCHESTRATOR_PROMPT)
    ])
    
    # Format the prompt with variables
    formatted_prompt = chat_prompt.format_messages(
        history_str=history_str, 
        input=state['input'], 
        image_flag=image_flag,
        profile=json.dumps(profile)
    )
    
    logging.debug(f"GEMINI: Prompt sent (length: {len(str(formatted_prompt))} chars)")
    try:
        # Invoke model with formatted prompt and request JSON output
        response = model.invoke(
            formatted_prompt,
            config={"response_format": {"type": "json_object"}}
        )
        raw_response = response.content.strip()
        logging.info(f"GEMINI: Raw response: {raw_response}")

        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = raw_response

        decision = json.loads(json_str)
        logging.info(f"GEMINI: Parsed decision: {decision}")

        state['conversation_stage'] = decision.get('stage', 'anamnesis')
        profile_update = decision.get('params', {}).get('profile_update', {})
        state['user_profile'] = {**profile, **profile_update}
        logging.info(f"FLOW: Profile updated: {json.dumps(profile_update)}")

        if decision['action'] in ['greet', 'question']:
            state['final_response'] = decision.get('response', 'Ceritakan keluhan gigi Anda?')
            state['sources'] = []  # No sources for non-RAG
            logging.info(f"FLOW: Action {decision['action']} - Response: {state['final_response'][:100]}...")
            return {"next": "end", "final_response": state['final_response'], "sources": []}
        elif decision['action'] == 'yolo':
            logging.info("FLOW: Routing to yolo/validator")
            return {"next": "validator"}
        elif decision['action'] == 'rag':
            logging.info("FLOW: Routing to rag/summarizer")
            return {"next": "summarizer"}
        else:
            logging.info(f"FLOW: Default end action")
            return {"next": "end",
                    "final_response": decision.get('response', 'Terima kasih! Konsultasikan ke dokter gigi ya.'),
                    "sources": []}

    except json.JSONDecodeError as je:
        logging.error(f"GEMINI: JSON parse error: {je}. Raw: {raw_response}")
        return {"next": "end", "final_response": "Maaf, saya perlu klarifikasi. Apa keluhan gigi utama Anda?",
                "sources": []}
    except Exception as e:
        logging.error(f"FLOW: Orchestrator error: {str(e)}")
        return {"next": "end", "final_response": "Error. Coba lagi dengan cerita keluhan gigi Anda.", "sources": []}


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
        "end": END
    }
)
graph.add_edge("validator", "detector")
graph.add_edge("detector", "summarizer")
graph.add_edge("summarizer", END)

app = graph.compile()


def run_agent(input_text, image_path=None, history=[]):
    logging.info(f"AGENT: Starting run - Input: {input_text}, Image: {image_path}, History len: {len(history)}")
    state = {
        "input": input_text,
        "image_path": image_path,
        "history": history,
        "conversation_stage": "greeting",
        "user_profile": {},
        "sources": []  # Initial
    }
    try:
        result = app.invoke(state)
        response = result.get('final_response', "Error occurred.")
        sources = result.get('sources', [])
        logging.info(f"AGENT: Run complete - Response length: {len(response)}, Sources: {len(sources)} (breakdown: PDF {len([s for s in sources if 'PDF' in s.get('source', '')])}, PubMed {len([s for s in sources if 'PubMed' in s.get('source', '')])})")
        return {'response': response, 'sources': sources}
    except Exception as e:
        logging.error(f"AGENT: Run failed: {str(e)}")
        return {'response': "Maaf, ada kesalahan teknis. Coba ulangi pertanyaan Anda.", 'sources': []}
