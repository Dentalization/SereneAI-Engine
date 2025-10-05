#!/usr/bin/env python3
"""RAG Ingestion Pipeline - Build and persist indices offline.

This script handles the INGESTION PHASE of the RAG system:
1. Load documents from PDF and PubMed
2. Apply semantic chunking
3. Generate embeddings
4. Build FAISS vectorstore
5. Build knowledge graph
6. Persist everything to disk

Usage:
    python scripts/build_indices.py [--enable-pubmed] [--docs-path PATH]

The runtime application (app.py) will then load pre-built indices for fast startup.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import setup_logging, load_config
from src.rag.rag_system import RAGSystem

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build RAG indices offline (ingestion pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build from local PDFs only
    python scripts/build_indices.py

    # Build from PDFs + PubMed
    python scripts/build_indices.py --enable-pubmed

    # Custom docs path
    python scripts/build_indices.py --docs-path /path/to/docs
        """
    )

    parser.add_argument(
        "--docs-path",
        type=str,
        default="docs/",
        help="Path to PDF documents directory (default: docs/)"
    )

    parser.add_argument(
        "--enable-pubmed",
        action="store_true",
        help="Fetch and include PubMed articles"
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="NeuML/pubmedbert-base-embeddings",
        help="Embedding model to use (default: pubmedbert-base-embeddings)"
    )

    parser.add_argument(
        "--index-dir",
        type=str,
        default="rag/faiss_index",
        help="Directory to save FAISS index (default: rag/faiss_index)"
    )

    parser.add_argument(
        "--kg-path",
        type=str,
        default="rag/kg.pkl",
        help="Path to save knowledge graph (default: rag/kg.pkl)"
    )

    return parser.parse_args()


def main():
    """Run ingestion pipeline."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging()

    logger.info("=" * 80)
    logger.info("RAG Ingestion Pipeline - Building Indices")
    logger.info("=" * 80)
    logger.info(f"Docs path: {args.docs_path}")
    logger.info(f"Enable PubMed: {args.enable_pubmed}")
    logger.info(f"Embedding model: {args.embedding_model}")
    logger.info(f"Index directory: {args.index_dir}")
    logger.info(f"KG path: {args.kg_path}")
    logger.info("=" * 80)

    # Check if docs directory exists
    docs_path = Path(args.docs_path)
    if not docs_path.exists():
        logger.error(f"Documents directory not found: {args.docs_path}")
        logger.error("Please create the directory and add PDF documents.")
        sys.exit(1)

    # Count PDF files
    pdf_files = list(docs_path.glob("**/*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {args.docs_path}")

    if len(pdf_files) == 0:
        logger.warning("No PDF files found. Continuing anyway...")

    # Initialize RAG system
    logger.info("Initializing RAG system...")
    rag = RAGSystem(
        docs_path=args.docs_path,
        enable_pubmed=args.enable_pubmed,
        embedding_model=args.embedding_model,
        index_dir=args.index_dir,
        kg_path=args.kg_path,
    )

    # Build indices (force rebuild)
    logger.info("Starting index building (this may take several minutes)...")
    start_time = time.time()

    try:
        rag.setup(force_rebuild=True)

        elapsed_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info("[OK] Ingestion Complete!")
        logger.info(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        logger.info(f"FAISS index saved to: {args.index_dir}")
        logger.info(f"Knowledge graph saved to: {args.kg_path}")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Run your application (app.py)")
        logger.info("2. It will automatically load the pre-built indices")
        logger.info("3. Enjoy fast startup time!")
        logger.info("")
        logger.info("To rebuild indices, run this script again.")

        return 0

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        logger.error("=" * 80)
        logger.error("Troubleshooting:")
        logger.error("- Check that PDF files exist in the docs directory")
        logger.error("- Ensure you have enough memory (embedding models can be large)")
        logger.error("- Check API keys if using PubMed")
        logger.error("- Review the full error traceback above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
