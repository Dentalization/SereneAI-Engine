"""Semantic chunking using sentence embeddings and clustering.

Instead of fixed-size chunks, this creates chunks based on semantic coherence:
- Group semantically similar sentences together
- Respect natural document boundaries
- Maintain context with overlapping boundaries
- Add provenance tracking for each chunk
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)


class ChunkMetadata(BaseModel):
    """Metadata for a semantic chunk."""

    chunk_id: str = Field(description="Unique chunk identifier")
    source_doc_id: str = Field(description="Original document ID")
    sentence_indices: List[int] = Field(description="Sentence positions in original doc")
    semantic_cluster: int = Field(description="Cluster ID this chunk belongs to")
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.now)
    chunk_size: int = Field(description="Number of characters")


class SemanticChunk(BaseModel):
    """A semantically coherent chunk of text."""

    content: str
    metadata: ChunkMetadata


class SemanticChunker:
    """Creates semantic chunks using sentence embeddings and clustering."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.7,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1500,
        overlap_sentences: int = 2,
    ):
        """Initialize semantic chunker.

        Args:
            model_name: Sentence transformer model for embeddings
            similarity_threshold: Threshold for sentence similarity (0-1)
            min_chunk_size: Minimum characters per chunk
            max_chunk_size: Maximum characters per chunk
            overlap_sentences: Number of sentences to overlap at boundaries
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences

        # Load model
        self.model: Optional[SentenceTransformer] = None
        self._load_model()

        logger.info(
            f"SemanticChunker: Initialized with {model_name}, "
            f"threshold={similarity_threshold}, size={min_chunk_size}-{max_chunk_size}"
        )

    def _load_model(self) -> None:
        """Lazy load sentence transformer."""
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"SemanticChunker: Loaded model {self.model_name}")
            except Exception as e:
                logger.error(f"SemanticChunker: Failed to load model - {e}")
                raise

    def chunk_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """Chunk documents semantically.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents with enhanced metadata
        """
        logger.info(f"SemanticChunker: Chunking {len(documents)} documents")

        all_chunks: List[Document] = []

        for doc_idx, doc in enumerate(documents):
            try:
                # Generate doc ID from content hash
                doc_id = self._generate_doc_id(doc.page_content, doc_idx)

                # Chunk this document
                doc_chunks = self._chunk_document(doc, doc_id)

                all_chunks.extend(doc_chunks)

                logger.debug(
                    f"SemanticChunker: Doc {doc_idx} → {len(doc_chunks)} chunks"
                )

            except Exception as e:
                logger.error(f"SemanticChunker: Failed to chunk doc {doc_idx} - {e}")
                # Fallback: use original doc as single chunk
                all_chunks.append(doc)

        logger.info(f"SemanticChunker: Created {len(all_chunks)} total chunks")
        return all_chunks

    def _chunk_document(
        self,
        doc: Document,
        doc_id: str
    ) -> List[Document]:
        """Chunk a single document semantically.

        Args:
            doc: Document to chunk
            doc_id: Unique document ID

        Returns:
            List of semantic chunks as Documents
        """
        # Split into sentences
        sentences = self._split_into_sentences(doc.page_content)

        if len(sentences) <= 3:
            # Too short to chunk meaningfully
            return [doc]

        logger.debug(f"SemanticChunker: Split doc into {len(sentences)} sentences")

        # Encode sentences
        embeddings = self.model.encode(
            sentences,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Cluster sentences by semantic similarity
        clusters = self._cluster_sentences(embeddings)

        logger.debug(f"SemanticChunker: Grouped into {max(clusters) + 1} clusters")

        # Build chunks from clusters
        chunks = self._build_chunks_from_clusters(
            sentences,
            clusters,
            doc,
            doc_id
        )

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting (could use spaCy/nltk for better accuracy)
        import re

        # Split on period, exclamation, question mark followed by space/newline
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Filter out very short sentences (likely artifacts)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        return sentences

    def _cluster_sentences(
        self,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """Cluster sentence embeddings by similarity.

        Args:
            embeddings: Sentence embedding matrix (n_sentences x embedding_dim)

        Returns:
            Cluster labels for each sentence
        """
        n_sentences = len(embeddings)

        # Determine number of clusters based on similarity
        # Use agglomerative clustering with distance threshold
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - self.similarity_threshold,  # Convert similarity to distance
            metric='cosine',
            linkage='average'
        )

        try:
            clusters = clustering.fit_predict(embeddings)
            return clusters

        except Exception as e:
            logger.warning(f"SemanticChunker: Clustering failed - {e}, using sequential groups")
            # Fallback: simple sequential grouping
            group_size = max(3, n_sentences // 5)  # ~5 groups
            return np.array([i // group_size for i in range(n_sentences)])

    def _build_chunks_from_clusters(
        self,
        sentences: List[str],
        clusters: np.ndarray,
        original_doc: Document,
        doc_id: str
    ) -> List[Document]:
        """Build chunks from clustered sentences.

        Args:
            sentences: Original sentences
            clusters: Cluster labels for each sentence
            original_doc: Original document for metadata
            doc_id: Document ID

        Returns:
            List of chunk Documents
        """
        chunks: List[Document] = []
        n_clusters = max(clusters) + 1

        for cluster_id in range(n_clusters):
            # Get sentences in this cluster
            cluster_sentence_indices = [
                i for i, c in enumerate(clusters) if c == cluster_id
            ]

            if not cluster_sentence_indices:
                continue

            # Add overlap with previous/next clusters
            start_idx = max(0, cluster_sentence_indices[0] - self.overlap_sentences)
            end_idx = min(len(sentences), cluster_sentence_indices[-1] + 1 + self.overlap_sentences)

            chunk_sentences = sentences[start_idx:end_idx]
            chunk_content = " ".join(chunk_sentences)

            # Check size constraints
            if len(chunk_content) < self.min_chunk_size:
                # Merge with next cluster
                continue

            if len(chunk_content) > self.max_chunk_size:
                # Split long chunk
                sub_chunks = self._split_long_chunk(chunk_content, cluster_sentence_indices)
                chunks.extend(sub_chunks)
            else:
                # Create chunk metadata
                chunk_id = self._generate_chunk_id(doc_id, cluster_id)

                chunk_metadata = original_doc.metadata.copy()
                chunk_metadata.update({
                    "chunk_id": chunk_id,
                    "source_doc_id": doc_id,
                    "sentence_indices": cluster_sentence_indices,
                    "semantic_cluster": cluster_id,
                    "chunk_size": len(chunk_content),
                    "created_at": datetime.now().isoformat(),
                })

                chunk_doc = Document(
                    page_content=chunk_content,
                    metadata=chunk_metadata
                )

                chunks.append(chunk_doc)

        logger.debug(f"SemanticChunker: Built {len(chunks)} chunks from {n_clusters} clusters")
        return chunks

    def _split_long_chunk(
        self,
        chunk_content: str,
        sentence_indices: List[int]
    ) -> List[Document]:
        """Split a chunk that's too long into smaller chunks.

        Args:
            chunk_content: Chunk text that's too long
            sentence_indices: Original sentence indices

        Returns:
            List of sub-chunks
        """
        # Simple approach: split at midpoint
        midpoint = len(chunk_content) // 2

        # Find nearest sentence boundary
        split_idx = chunk_content[:midpoint].rfind(". ") + 1
        if split_idx <= 0:
            split_idx = midpoint

        part1 = chunk_content[:split_idx].strip()
        part2 = chunk_content[split_idx:].strip()

        # Create documents (simplified metadata)
        chunks = []
        if len(part1) >= self.min_chunk_size:
            chunks.append(Document(page_content=part1, metadata={"split": "part1"}))
        if len(part2) >= self.min_chunk_size:
            chunks.append(Document(page_content=part2, metadata={"split": "part2"}))

        return chunks

    def _generate_doc_id(self, content: str, index: int) -> str:
        """Generate unique document ID.

        Args:
            content: Document content
            index: Document index

        Returns:
            Unique ID
        """
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"doc_{index}_{content_hash}"

    def _generate_chunk_id(self, doc_id: str, cluster_id: int) -> str:
        """Generate unique chunk ID.

        Args:
            doc_id: Parent document ID
            cluster_id: Cluster number

        Returns:
            Unique chunk ID
        """
        return f"{doc_id}_chunk_{cluster_id}"