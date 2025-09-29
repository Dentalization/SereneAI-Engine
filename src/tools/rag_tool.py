# rag_tool.py
from langchain_community.document_loaders import PyPDFDirectoryLoader, PubMedLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.prompts import PromptTemplate
from cachetools import LRUCache
from src.config import load_config
import os
import logging
import importlib
import networkx as nx
import json
from typing import List, Tuple, Dict

config = load_config()
cache = LRUCache(maxsize=config['cache_size'])

vectorstore = None
retriever = None
knowledge_graph = None

KG_TRIPLE_DELIMITER = " | "

_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE = (
    "You are a networked intelligence helping track knowledge triples about dental conditions, treatments, etc."
    " Extract triples (subject, predicate, object) from text. Focus on medical relations like 'caries causes pain' or 'fluoride prevents caries'.\n\n"
    f"EXAMPLE\nText: Caries is caused by bacteria and prevented by fluoride.\n"
    f"Output: (caries, is caused by, bacteria){KG_TRIPLE_DELIMITER}(fluoride, prevents, caries)\n"
    "END OF EXAMPLE\n\n"
    "{text}\nOutput:"
)

KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE,
)

# RAG prompt: Dental-focused, empathetic, actionable
DENTAL_RAG_PROMPT = (
    "You are a helpful dental assistant. Based on retrieved context: {context}\n"
    "Detections: {detections}, Spatial: {spatial_insights}, Graph: {graph_results}.\n"
    "User profile/history: {profile}.\n"
    "Analyze query: {query}. Provide causes, treatments, prevention. "
    "Respond in user's language (Indo if query in Indo). Actionable, empathetic, evidence-based. "
    "If no info, use general knowledge (e.g., caries: bacteria cause, fillings treat, fluoride prevents)."
)


def parse_triples(response: str) -> List[Tuple[str, str, str]]:
    if not response:
        return []
    triples = [t.strip() for t in response.split(KG_TRIPLE_DELIMITER) if t.strip()]
    parsed = []
    for triple in triples:
        parts = [p.strip().strip('()') for p in triple.split(',')]
        if len(parts) == 3:
            parsed.append(tuple(parts))
    return parsed


def create_graph_from_triples(triples: List[Tuple[str, str, str]]) -> nx.DiGraph:
    G = nx.DiGraph()
    for subject, predicate, obj in triples:
        G.add_edge(subject.lower(), obj.lower(), label=predicate.lower())
    logging.info(f"RAG: KG built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def query_graph(query: str, graph: nx.DiGraph) -> str:
    logging.debug(f"RAG: Querying graph for '{query}'")
    query_lower = query.lower()
    results = []
    for node in graph.nodes():
        if query_lower in node:
            neighbors = list(graph.neighbors(node))
            if neighbors:
                results.append(f"{node} relates to: {', '.join(neighbors)} via {graph[node][neighbors[0]]['label']}")
            if len(neighbors) > 0:
                try:
                    paths = nx.shortest_path(graph, node, neighbors[0])
                    results.append(f"Path: {' -> '.join(paths)}")
                except nx.NetworkXNoPath:
                    pass
    graph_res = "; ".join(results) if results else "No relational insights found."
    logging.debug(f"RAG: Graph results: {graph_res[:100]}...")
    return graph_res


def setup_rag():
    global vectorstore, retriever, knowledge_graph
    if vectorstore is None:
        logging.info("RAG: Starting setup...")
        loader = PyPDFDirectoryLoader(config['docs_path'])
        docs = loader.load()
        # Enhanced metadata for PDF (page, filename)
        for doc in docs:
            doc.metadata['source'] = doc.metadata.get('source', 'PDF')  # Filename from source
            doc.metadata['page'] = doc.metadata.get('page', 1)
            doc.metadata['title'] = os.path.basename(doc.metadata.get('source', 'Unknown'))
        logging.info(f"RAG: Loaded {len(docs)} PDF docs with page metadata")
        try:
            pubmed_loader = PubMedLoader("dental conditions caries gingivitis Indonesia")
            pubmed_docs = pubmed_loader.load()
            # PubMed (sama, enhanced)
            if pubmed_docs:
                for i, doc in enumerate(pubmed_docs):
                    doc.metadata['source'] = f"PubMed_{i + 1}"
                    doc.metadata['title'] = doc.metadata.get('Title', 'PubMed Article')
                    doc.metadata['authors'] = doc.metadata.get('Authors', 'Unknown')
                    doc.metadata['pmid'] = doc.metadata.get('PMID', 'N/A')
                docs.extend(pubmed_docs)
                logging.info(f"RAG: Loaded {len(pubmed_docs)} PubMed docs with metadata")
            else:
                logging.warning("RAG: No PubMed docs; using PDF only.")
        except Exception as e:
            logging.warning(f"RAG: PubMed load error: {e}. Using PDF only.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_documents(docs)
        logging.info(f"RAG: Split into {len(texts)} chunks")
        try:
            embeddings = HuggingFaceEmbeddings(model_name="dmis-lab/biobert-base-cased-v1.1")
            logging.info("RAG: BioBERT embeddings loaded successfully.")
        except Exception as e:
            logging.warning(f"RAG: BioBERT fallback: {e}.")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(texts, embeddings)
        semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        bm25_retriever = None
        if importlib.util.find_spec("rank_bm25"):
            bm25_retriever = BM25Retriever.from_documents(texts)
        if bm25_retriever:
            ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, semantic_retriever], weights=[0.4, 0.6])
        else:
            ensemble_retriever = semantic_retriever
        cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)
        retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config['gemini_api_key'], temperature=0)
        # Create chain for triple extraction
        from langchain_core.output_parsers import StrOutputParser
        triple_chain = KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT | llm | StrOutputParser()
        all_triples = []
        for text_doc in texts[:10]:
            try:
                triples_str = triple_chain.invoke({'text': text_doc.page_content})
                all_triples.extend(parse_triples(triples_str))
            except Exception as e:
                logging.error(f"RAG: Triple extraction error: {e}")
        knowledge_graph = create_graph_from_triples(all_triples)
        logging.info("RAG: Setup complete")


def route_query(query: str, detections: str = '', spatial_insights: str = '') -> str:
    logging.debug(f"RAG: Routing query '{query}' with detections/spatial")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config['gemini_api_key'], temperature=0)
    from langchain_core.messages import HumanMessage
    route_prompt = f"Classify '{query}' (detections: {detections}, spatial: {spatial_insights}). Output 'graph' if relational (causes/treats/prevention), 'vector' if definition/facts."
    try:
        response = llm.invoke([HumanMessage(content=route_prompt)])
        decision = response.content.strip().lower()
        logging.info(f"RAG: Routing decision: {decision}")
        return "graph" if "graph" in decision else "vector"
    except Exception as e:
        logging.error(f"RAG: Routing error: {e}. Fallback to vector.")
        return "vector"


def query_rag(query: str, detections='', spatial_insights='', history: List[dict] = [], profile: dict = {}) -> tuple[
    str, List[Dict]]:  # Return response + sources
    logging.info(f"RAG: Starting query - '{query}', Profile: {json.dumps(profile)[:150]}...")
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-3:]])
    profile_str = json.dumps(profile)
    cache_key = f"{query}_{detections}_{spatial_insights}_{hash(history_str)}_{hash(profile_str)}_graph"
    if cache_key in cache:
        logging.debug(f"RAG: Cache hit for key {cache_key[:50]}...")
        return cache[cache_key][0], cache[cache_key][1]  # (response, sources)

    setup_rag()
    route = route_query(query, detections, spatial_insights)

    # Manual retrieval with metadata
    try:
        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs])
        # Extract sources metadata
        sources = []
        for i, doc in enumerate(docs):
            source_info = {
                "id": i + 1,
                "title": doc.metadata.get('title', 'Unknown'),
                "source": doc.metadata.get('source', 'Unknown'),
                "page": doc.metadata.get('page', 'N/A') if 'PDF' in doc.metadata.get('source', '') else 'N/A',
                "authors": doc.metadata.get('authors', '') if 'PubMed' in doc.metadata.get('source', '') else '',
                "pmid": doc.metadata.get('pmid', '') if 'PubMed' in doc.metadata.get('source', '') else '',
                "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources.append(source_info)
        pdf_count = len([s for s in sources if 'PDF' in s['source']])
        pubmed_count = len([s for s in sources if 'PubMed' in s['source']])
        logging.info(f"RAG: Retrieved {len(docs)} docs (PDF: {pdf_count}, PubMed: {pubmed_count})")
        for src in sources:
            logging.debug(
                f"RAG: Source {src['id']}: {src['title']} ({src['source']}) - Snippet: {src['snippet'][:100]}...")
    except Exception as e:
        logging.error(f"RAG: Retrieval error: {e}")
        context = "No context available."
        sources = []

    graph_results = ""
    if route == "graph" and knowledge_graph:
        graph_results = query_graph(query, knowledge_graph)

    # Format prompt with all vars
    full_prompt = DENTAL_RAG_PROMPT.format(
        context=context, query=query, detections=detections,
        spatial_insights=spatial_insights, graph_results=graph_results, profile=profile_str
    )
    logging.debug(f"RAG: Full prompt length: {len(full_prompt)} chars")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config['gemini_api_key'], temperature=0.3)
    from langchain_core.messages import HumanMessage
    try:
        response = llm.invoke([HumanMessage(content=full_prompt)])
        rag_result = response.content
        logging.info(f"RAG: LLM response generated (length: {len(rag_result)} chars, snippet: {rag_result[:100]}...)")
        # Fallback if empty/short
        if not rag_result.strip() or len(rag_result) < 50:
            logging.warning("RAG: Response too short, using fallback")
            fallback_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config['gemini_api_key'],
                                              temperature=0.7)
            fallback_prompt = f"Based on profile {profile_str} and query '{query}', provide empathetic dental advice: causes, treatments, prevention. Use general knowledge if needed."
            fallback_response = fallback_llm.invoke([HumanMessage(content=fallback_prompt)])
            rag_result = fallback_response.content
        full_response = f"{rag_result}\nGraph Relations: {graph_results}" if graph_results else rag_result
        cache[cache_key] = (full_response, sources)  # Cache with sources
        logging.info(f"RAG: Final response ready (length: {len(full_response)})")
        return full_response, sources
    except Exception as e:
        logging.error(f"RAG: LLM error: {str(e)}")
        fallback_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config['gemini_api_key'])
        fallback_response = fallback_llm.invoke([HumanMessage(
            content=f"Berikan saran dental empati untuk query: {query}. Sertakan causes, treatments, prevention berdasarkan profile: {profile_str}.")])
        fallback = fallback_response.content
        logging.info(f"RAG: Fallback response used (snippet: {fallback[:100]}...)")
        return fallback, []
