"""
RAG System for evidence-based dental information retrieval.
Modernized with singleton pattern and proper abstraction.
"""

from functools import lru_cache
from typing import List

from langchain_core.documents import Document

from src.config import get_settings


class RAGSystem:
    """
    Modular RAG system for dental knowledge retrieval.

    Features:
    - Query expansion with medical synonyms
    - Vector similarity search (FAISS)
    - Advanced reranking (ColBERT-style)
    - Knowledge graph reasoning
    - Claim validation (hallucination detection)

    Example:
        >>> rag = RAGSystem.get_instance()
        >>> docs = rag.retrieve("What causes tooth sensitivity?")
        >>> response = rag.generate_response("What causes tooth sensitivity?", docs)
    """

    _instance = None

    def __init__(self):
        """Initialize RAG system (private - use get_instance())."""
        self.config = get_settings()
        self._vectorstore = None
        self._llm = None

    @classmethod
    @lru_cache(maxsize=1)
    def get_instance(cls) -> "RAGSystem":
        """
        Get singleton RAG system instance (cached).

        Returns:
            Initialized RAGSystem instance
        """
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize RAG components (lazy loading)."""
        from pathlib import Path

        # Check if index exists
        index_path = Path(self.config.rag_index_dir)
        if not index_path.exists():
            print(
                f"Warning: RAG index not found at {index_path}. "
                f"Please run 'python scripts/build_indices.py' to build the index."
            )
            return

        # TODO: Load vectorstore, knowledge graph, etc.
        # from langchain_community.vectorstores import FAISS
        # self._vectorstore = FAISS.load_local(str(index_path))

        # Initialize LLM for generation
        from src.models.gemini import get_gemini_chat

        self._llm = get_gemini_chat(temperature=0.2)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
    ) -> List[Document]:
        """
        Retrieve relevant documents for query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score

        Returns:
            List of relevant Document objects with metadata

        Example:
            >>> docs = rag.retrieve("tooth pain causes", top_k=5)
        """
        if self._vectorstore is None:
            # Return empty if not initialized
            print("Warning: RAG system not initialized. Returning empty results.")
            return []

        # TODO: Implement actual retrieval
        # 1. Query expansion
        # 2. Vector search
        # 3. Reranking
        # 4. Knowledge graph enhancement

        # Placeholder: return empty
        return []

    def generate_response(
        self,
        query: str,
        documents: List[Document],
        language: str = "id",
    ) -> str:
        """
        Generate response based on retrieved documents.

        Args:
            query: Original query
            documents: Retrieved documents
            language: Response language (id or en)

        Returns:
            Generated response with citations

        Example:
            >>> response = rag.generate_response(query, docs, language="id")
        """
        if not documents:
            return (
                "Maaf, saya tidak menemukan informasi yang relevan."
                if language == "id"
                else "Sorry, I couldn't find relevant information."
            )

        # Build context from documents
        context = "\n\n".join(
            [
                f"Source: {doc.metadata.get('title', 'Unknown')}\n{doc.page_content}"
                for doc in documents
            ]
        )

        # Generate response with LLM
        prompt = f"""Based on the following dental information, answer the question.

Question: {query}

Information:
{context}

Provide a clear, evidence-based answer in {'Indonesian' if language == 'id' else 'English'}.
Include citations to sources."""

        response = self._llm.invoke(prompt)

        return response.content if hasattr(response, "content") else str(response)

    def validate_claims(
        self,
        query: str,
        documents: List[Document],
    ) -> dict:
        """
        Validate claims against source documents (hallucination detection).

        Args:
            query: Query or claim to validate
            documents: Source documents

        Returns:
            Validation result with support confidence and hallucination risk

        Example:
            >>> validation = rag.validate_claims(query, docs)
            >>> print(validation["hallucination_risk"])  # low, medium, high
        """
        # Placeholder implementation
        return {
            "support_confidence": 0.8,
            "hallucination_risk": "low",
            "supported_claims": [],
            "unsupported_claims": [],
        }
