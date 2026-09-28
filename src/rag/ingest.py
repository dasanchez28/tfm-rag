"""
Corpus ingestion and indexing in the vector database (ChromaDB).

Builds the dense (embedding) index. The sparse (BM25) index is built in memory
in retrievers.py from the same chunks.

Usage:
    python -m src.rag.ingest
"""

from __future__ import annotations

import json

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.config import CONFIG, PROCESSED_DIR, ROOT
from src.rag.llm import get_embeddings


def load_chunks() -> list[Document]:
    """Load the processed chunks as LangChain documents."""
    with open(PROCESSED_DIR / "chunks.json", encoding="utf-8") as f:
        raw = json.load(f)
    return [
        Document(
            page_content=c["text"],
            metadata={
                "chunk_id": c["chunk_id"],
                "arxiv_id": c["arxiv_id"],
                "title": c["title"],
                "source": c["source"],
            },
        )
        for c in raw
    ]


def build_index() -> Chroma:
    """Embed all chunks and persist the vector index."""
    persist_dir = str(ROOT / CONFIG["vector_store"]["persist_dir"])
    collection = CONFIG["vector_store"]["collection_name"]

    docs = load_chunks()
    print(f"[Ingest] Indexing {len(docs)} chunks in ChromaDB ...")
    vs = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=collection,
        persist_directory=persist_dir,
    )
    print(f"[Ingest] Index persisted in {persist_dir}")
    return vs


def load_index() -> Chroma:
    """Load the persisted index (without re-indexing)."""
    persist_dir = str(ROOT / CONFIG["vector_store"]["persist_dir"])
    collection = CONFIG["vector_store"]["collection_name"]
    return Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=persist_dir,
    )


if __name__ == "__main__":
    build_index()
