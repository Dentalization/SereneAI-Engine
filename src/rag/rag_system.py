"""Advanced Retrieval-Augmented Generation system.

Integrates:
- Query expansion (medical synonyms)
- Semantic chunking (instead of fixed-size)
- Advanced reranking (ColBERT-style + temporal)
- Knowledge graph reasoning
- Claim validation with citation tracing
"""
from __future__ import annotations

import logging
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
    """Advanced RAG system with all enhancements."""

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
            docs_path: Directory with PDF documents
            enable_pubmed: Whether to fetch PubMed articles
            embedding_model: Model for embeddings
            index_dir: Directory to save/load FAISS index
            kg_path: Path to save/load knowledge graph
        """
        self.docs_path = docs_path
        self.enable_pubmed = enable_pubmed
        self.embedding_model_name = embedding_model
        self.index_dir = index_dir
        self.kg_path = kg_path

        # Components (lazy initialized)
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.vectorstore: Optional[FAISS] = None
        self.kg_builder: Optional[KnowledgeGraphBuilder] = None

        # Modules
        self.query_expander = QueryExpander()
        self.semantic_chunker = SemanticChunker()
        self.reranker = AdvancedReranker()
        self.claim_validator = ClaimValidator()

        # LLM for generation
        self.llm = get_gemini_chat(model="gemini-2.5-flash", temperature=0.3)

        logger.info("RAGSystem: Initialized")

    def setup(self, force_rebuild: bool = False) -> None:
        """Setup RAG system (load or build indices).

        Args:
            force_rebuild: Whether to force rebuild even if indices exist
        """
        logger.info("RAGSystem: Starting setup...")

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        # Try load existing
        if not force_rebuild:
            loaded = self._try_load_existing()
            if loaded:
                logger.info("RAGSystem: Loaded existing indices")
                return

        # Build from scratch
        logger.info("RAGSystem: Building indices from scratch...")

        # Load documents
        docs = self._load_documents()
        logger.info(f"RAGSystem: Loaded {len(docs)} raw documents")

        # Semantic chunking
        chunks = self.semantic_chunker.chunk_documents(docs)
        logger.info(f"RAGSystem: Created {len(chunks)} semantic chunks")

        # Build vector store
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        logger.info("RAGSystem: Built FAISS vectorstore")

        # Build knowledge graph
        self.kg_builder = KnowledgeGraphBuilder()
        self.kg_builder.build_from_documents(docs, persist_path=self.kg_path)
        logger.info("RAGSystem: Built knowledge graph")

        # Persist
        self._persist_indices()

        logger.info("RAGSystem: Setup complete")

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

        # Ensure setup
        if self.vectorstore is None or self.kg_builder is None:
            self.setup()

        # Step 1: Query expansion
        expanded = self.query_expander.expand(query)
        logger.info(f"RAGSystem: Expanded query - {len(expanded.synonyms)} synonyms")

        # Step 2: Vector retrieval (use expanded query)
        retrieval_query = expanded.expanded_query
        retrieved_docs = self.vectorstore.similarity_search(
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
        kg_result = self.kg_builder.query(query, max_hops=3)
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
                pubmed_loader = PubMedLoader("dental caries gingivitis Indonesia")
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
        """Try to load existing indices.

        Returns:
            True if loaded successfully
        """
        try:
            import os

            if os.path.isdir(self.index_dir) and os.path.exists(self.kg_path):
                # Load vectorstore
                self.vectorstore = FAISS.load_local(
                    self.index_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )

                # Load KG
                self.kg_builder = KnowledgeGraphBuilder.load(self.kg_path)

                return True

        except Exception as e:
            logger.warning(f"RAGSystem: Load failed - {e}")

        return False

    def _persist_indices(self) -> None:
        """Persist vectorstore and KG to disk."""
        try:
            if self.vectorstore:
                self.vectorstore.save_local(self.index_dir)
                logger.info(f"RAGSystem: Saved vectorstore to {self.index_dir}")

            if self.kg_builder:
                self.kg_builder.save(self.kg_path)
                logger.info(f"RAGSystem: Saved KG to {self.kg_path}")

        except Exception as e:
            logger.error(f"RAGSystem: Persist failed - {e}")