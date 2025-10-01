"""Advanced Retrieval-Augmented Generation system.

Integrates:
- Query expansion (medical synonyms)
- Semantic chunking (threshold-based, optimized)
- Advanced reranking (ColBERT-style + temporal)
- Knowledge graph reasoning
- Claim validation with citation tracing

Architecture:
- Two-phase design: Ingestion (offline) vs Runtime (online)
- Ingestion: Build indices, save to disk (scripts/build_indices.py)
- Runtime: Load pre-built indices from disk (this file)
- Centralized embedding models for efficiency
- Lazy loading of heavy components
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.document_loaders import PyPDFDirectoryLoader, PubMedLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

from src.agents.state_models import SourceCitation
from src.config import load_config
from src.rag.advanced_reranker import AdvancedReranker
from src.rag.claim_validator import ClaimValidator
from src.rag.knowledge_graph import KnowledgeGraphBuilder
from src.rag.query_expander import QueryExpander
from src.rag.semantic_chunker import SemanticChunker
from src.utils.embeddings import get_langchain_embeddings
from src.utils.llm import get_gemini_chat

logger = logging.getLogger(__name__)

config = load_config()


class RAGResult(BaseModel):
    """RAG retrieval and generation result."""

    response: str
    sources: List[SourceCitation]
    validation_result: Optional[Dict[str, Any]] = None
    kg_insights: str = ""
    expanded_query: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)


class RAGSystem:
    """Advanced RAG system with optimized two-phase architecture.

    Ingestion Phase (offline - scripts/build_indices.py):
    - Load documents
    - Semantic chunking
    - Generate embeddings
    - Build FAISS index
    - Build knowledge graph
    - Persist to disk

    Runtime Phase (online - this class in query mode):
    - Load pre-built indices from disk
    - Lazy-load heavy components
    - Fast startup
    - Real-time query processing
    """

    def __init__(
        self,
        docs_path: str = "docs/",
        enable_pubmed: bool = False,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_dir: str = ".rag/faiss_index",
        kg_path: str = ".rag/kg.pkl",
    ):
        """Initialize RAG system.

        Args:
            docs_path: Directory with PDF documents (used during ingestion)
            enable_pubmed: Whether to fetch PubMed articles (used during ingestion)
            embedding_model: Model for embeddings
            index_dir: Directory to save/load FAISS index
            kg_path: Path to save/load knowledge graph
        """
        self.docs_path = docs_path
        self.enable_pubmed = enable_pubmed
        self.embedding_model_name = embedding_model
        self.index_dir = index_dir
        self.kg_path = kg_path

        # Heavy components (lazy initialized for fast startup)
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._vectorstore: Optional[FAISS] = None
        self._kg_builder: Optional[KnowledgeGraphBuilder] = None
        self._semantic_chunker: Optional[SemanticChunker] = None

        # Lightweight modules (initialized immediately)
        self.query_expander = QueryExpander()
        self.reranker = AdvancedReranker()
        self.claim_validator = ClaimValidator()

        # LLM for generation (API client - lightweight)
        self.llm = get_gemini_chat(model="gemini-2.5-flash", temperature=0.3)

        logger.info("RAGSystem: Initialized (components will be lazy-loaded)")

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy-load embeddings model (shared singleton)."""
        if self._embeddings is None:
            self._embeddings = get_langchain_embeddings(self.embedding_model_name)
            logger.info(f"RAGSystem: Loaded embeddings model")
        return self._embeddings

    @property
    def vectorstore(self) -> Optional[FAISS]:
        """Get vectorstore (may be None until setup() is called)."""
        return self._vectorstore

    @property
    def kg_builder(self) -> Optional[KnowledgeGraphBuilder]:
        """Get knowledge graph builder (may be None until setup() is called)."""
        return self._kg_builder

    @property
    def semantic_chunker(self) -> SemanticChunker:
        """Lazy-load semantic chunker (only needed during ingestion)."""
        if self._semantic_chunker is None:
            self._semantic_chunker = SemanticChunker(
                model_name=self.embedding_model_name,
                similarity_threshold=0.75,  # Optimized for medical content
                min_chunk_size=150,
                max_chunk_size=1200,
            )
            logger.info(f"RAGSystem: Loaded semantic chunker")
        return self._semantic_chunker

    def setup(self, force_rebuild: bool = False) -> None:
        """Setup RAG system (load or build indices).

        Best Practice:
        - In production/runtime: Call with force_rebuild=False (default)
          → Fast startup, just loads from disk
        - In ingestion pipeline: Call with force_rebuild=True
          → Rebuilds indices from source documents

        Args:
            force_rebuild: Whether to force rebuild even if indices exist
        """
        logger.info("RAGSystem: Starting setup...")

        # Try load existing (fast path for runtime)
        if not force_rebuild:
            loaded = self._try_load_existing()
            if loaded:
                logger.info("RAGSystem: [OK] Loaded existing indices (fast startup)")
                return
            else:
                logger.warning(
                    "RAGSystem: No existing indices found. "
                    "Run scripts/build_indices.py first or call setup(force_rebuild=True)"
                )

        # Build from scratch (slow path for ingestion)
        logger.info("RAGSystem: Building indices from scratch (this will take a while)...")

        # Load documents
        docs = self._load_documents()
        logger.info(f"RAGSystem: Loaded {len(docs)} raw documents")

        # Semantic chunking (uses shared embedding model)
        chunks = self.semantic_chunker.chunk_documents(docs)
        logger.info(f"RAGSystem: Created {len(chunks)} semantic chunks")

        # Build vector store (uses shared embedding model)
        self._vectorstore = FAISS.from_documents(chunks, self.embeddings)
        logger.info("RAGSystem: Built FAISS vectorstore")

        # Build knowledge graph
        self._kg_builder = KnowledgeGraphBuilder()
        self._kg_builder.build_from_documents(docs, persist_path=self.kg_path)
        logger.info("RAGSystem: Built knowledge graph")

        # Persist with metadata
        self._persist_indices(num_docs=len(docs), num_chunks=len(chunks))

        logger.info("RAGSystem: [OK] Setup complete")

    def query(
        self,
        query: str,
        detections: str = "",
        spatial_insights: str = "",
        history: Optional[List[dict]] = None,
        profile: Optional[dict] = None,
        top_k: int = 5,
    ) -> RAGResult:
        """Query RAG system with all enhancements.

        Args:
            query: User query
            detections: YOLO detections JSON
            spatial_insights: Spatial analysis from vision
            history: Conversation history
            profile: User profile
            top_k: Number of sources to retrieve

        Returns:
            RAGResult with response and metadata
        """
        logger.info(f"RAGSystem: Query - '{query[:50]}...'")

        # Ensure indices are loaded
        if self._vectorstore is None or self._kg_builder is None:
            logger.info("RAGSystem: Indices not loaded, loading now...")
            self.setup(force_rebuild=False)

        # Step 1: Query expansion
        expanded = self.query_expander.expand(query)
        logger.info(f"RAGSystem: Expanded query - {len(expanded.synonyms)} synonyms")

        # Step 2: Vector retrieval (use expanded query)
        retrieval_query = expanded.expanded_query
        retrieved_docs = self._vectorstore.similarity_search(
            retrieval_query,
            k=top_k * 2  # Retrieve more for reranking
        )
        logger.info(f"RAGSystem: Retrieved {len(retrieved_docs)} documents")

        # Step 3: Rerank with advanced reranker
        reranked_docs = self.reranker.rerank(
            query=query,  # Use original query for reranking
            documents=retrieved_docs,
            top_k=top_k
        )
        logger.info(f"RAGSystem: Reranked to top {len(reranked_docs)}")

        # Step 4: Knowledge graph query
        kg_result = self._kg_builder.query(query, max_hops=3)
        kg_insights = kg_result.get("insights", "")
        logger.info(f"RAGSystem: KG insights - {len(kg_insights)} chars")

        # Step 5: Generate response
        response = self._generate_response(
            query=query,
            documents=reranked_docs,
            kg_insights=kg_insights,
            detections=detections,
            spatial_insights=spatial_insights,
            profile=profile or {}
        )
        logger.info(f"RAGSystem: Generated response - {len(response)} chars")

        # Step 6: Validate claims
        validation = self.claim_validator.validate(
            response=response,
            sources=reranked_docs
        )
        logger.info(
            f"RAGSystem: Validation - Confidence={validation.overall_confidence:.2f}, "
            f"Risk={validation.hallucination_risk}"
        )

        # Step 7: Build source citations
        sources = self._build_citations(reranked_docs)

        # Return result
        result = RAGResult(
            response=response,
            sources=sources,
            validation_result=validation.model_dump(),
            kg_insights=kg_insights,
            expanded_query=expanded.expanded_query,
            confidence=validation.overall_confidence
        )

        return result

    def _load_documents(self) -> List[Document]:
        """Load documents from PDF and optionally PubMed.

        Returns:
            List of documents
        """
        docs = []

        # Load PDFs
        loader = PyPDFDirectoryLoader(self.docs_path)
        pdf_docs = loader.load()

        for doc in pdf_docs:
            doc.metadata["provider"] = "PDF"
            doc.metadata["title"] = doc.metadata.get("source", "Unknown").split("/")[-1]

        docs.extend(pdf_docs)
        logger.info(f"RAGSystem: Loaded {len(pdf_docs)} PDF documents")

        # Load PubMed
        if self.enable_pubmed:
            try:
                pubmed_loader = PubMedLoader("dental caries gingivitis")
                pubmed_docs = pubmed_loader.load()

                for doc in pubmed_docs:
                    doc.metadata["provider"] = "PubMed"
                    doc.metadata["title"] = doc.metadata.get("Title", "PubMed Article")

                docs.extend(pubmed_docs)
                logger.info(f"RAGSystem: Loaded {len(pubmed_docs)} PubMed documents")

            except Exception as e:
                logger.warning(f"RAGSystem: PubMed load failed - {e}")

        return docs

    def _generate_response(
        self,
        query: str,
        documents: List[Document],
        kg_insights: str,
        detections: str,
        spatial_insights: str,
        profile: dict
    ) -> str:
        """Generate response using retrieved context.

        Args:
            query: User query
            documents: Retrieved documents
            kg_insights: Knowledge graph insights
            detections: Detection results
            spatial_insights: Spatial analysis
            profile: User profile

        Returns:
            Generated response text
        """
        # Build context
        context = "\n\n".join([
            f"[Source {i+1}] {doc.page_content[:500]}"
            for i, doc in enumerate(documents)
        ])

        # Build prompt
        prompt = f"""You are a helpful dental assistant. Provide evidence-based advice.

**Context from sources:**
{context}

**Knowledge Graph Insights:**
{kg_insights}

**User Query:** {query}

**Detections:** {detections}
**Spatial Analysis:** {spatial_insights}
**User Profile:** {profile}

**Instructions:**
- Base your answer on the provided context
- Cite sources when making claims
- Be empathetic and actionable
- Respond in Indonesian if query is in Indonesian
- Recommend dentist visit for serious issues

**Response:**"""

        from langchain_core.messages import HumanMessage

        response_obj = self.llm.invoke([HumanMessage(content=prompt)])
        response = response_obj.content

        return response

    def _build_citations(self, documents: List[Document]) -> List[SourceCitation]:
        """Build source citations from documents.

        Args:
            documents: Retrieved documents

        Returns:
            List of SourceCitation objects
        """
        citations = []

        for i, doc in enumerate(documents):
            provider = doc.metadata.get("provider", "PDF")
            citation = SourceCitation(
                id=i + 1,
                title=doc.metadata.get("title", "Unknown"),
                provider=provider,
                snippet=doc.page_content[:200],
                source_path=doc.metadata.get("source", ""),
                page=doc.metadata.get("page"),
                confidence=doc.metadata.get("relevance_score", 1.0)
            )

            if provider == "PubMed":
                citation.pmid = doc.metadata.get("pmid", "")
                citation.url = f"https://pubmed.ncbi.nlm.nih.gov/{citation.pmid}/" if citation.pmid else ""
                citation.authors = doc.metadata.get("authors", "")

            citations.append(citation)

        return citations

    def _try_load_existing(self) -> bool:
        """Try to load existing indices from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if os.path.isdir(self.index_dir) and os.path.exists(self.kg_path):
                # Load vectorstore
                self._vectorstore = FAISS.load_local(
                    self.index_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )

                # Load KG
                self._kg_builder = KnowledgeGraphBuilder.load(self.kg_path)

                # Load metadata if exists
                metadata_path = os.path.join(self.index_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                        logger.info(
                            f"RAGSystem: Loaded indices - "
                            f"Version: {metadata.get('version', 'unknown')}, "
                            f"Created: {metadata.get('created_at', 'unknown')}, "
                            f"Docs: {metadata.get('num_documents', '?')}, "
                            f"Chunks: {metadata.get('num_chunks', '?')}"
                        )

                return True

        except Exception as e:
            logger.warning(f"RAGSystem: Load failed - {e}")

        return False

    def _persist_indices(self, num_docs: int = 0, num_chunks: int = 0) -> None:
        """Persist vectorstore and KG to disk with metadata.

        Args:
            num_docs: Number of source documents
            num_chunks: Number of semantic chunks created
        """
        try:
            # Ensure directory exists
            os.makedirs(self.index_dir, exist_ok=True)

            # Save vectorstore
            if self._vectorstore:
                self._vectorstore.save_local(self.index_dir)
                logger.info(f"RAGSystem: Saved vectorstore to {self.index_dir}")

            # Save KG
            if self._kg_builder:
                self._kg_builder.save(self.kg_path)
                logger.info(f"RAGSystem: Saved KG to {self.kg_path}")

            # Save metadata
            metadata = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "num_documents": num_docs,
                "num_chunks": num_chunks,
                "embedding_model": self.embedding_model_name,
                "index_type": "FAISS",
                "kg_path": self.kg_path,
            }
            metadata_path = os.path.join(self.index_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"RAGSystem: Saved metadata to {metadata_path}")

        except Exception as e:
            logger.error(f"RAGSystem: Persist failed - {e}")