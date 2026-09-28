"""
Retrievers: dense (vector), sparse (BM25) and hybrid (Reciprocal Rank Fusion).

The dense and sparse rankings are fused with Reciprocal Rank Fusion (RRF),
which only uses rank positions and is therefore robust to the different
score scales of cosine similarity and BM25.
"""

from __future__ import annotations

import json
import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config import CONFIG, PROCESSED_DIR
from src.rag.ingest import load_index


def _tokenize(text: str) -> list[str]:
    """Lower-case word tokenizer (also keeps common accented Latin letters)."""
    return re.findall(r"[a-záéíóúñ0-9]+", text.lower())


class DenseRetriever:
    """Semantic retrieval over the vector index (ChromaDB)."""

    def __init__(self):
        self.vs = load_index()

    def retrieve(self, query: str, k: int) -> list[Document]:
        return self.vs.similarity_search(query, k=k)


class BM25Retriever:
    """Sparse lexical retrieval with BM25."""

    def __init__(self):
        with open(PROCESSED_DIR / "chunks.json", encoding="utf-8") as f:
            raw = json.load(f)
        self.docs = [
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
        self._bm25 = BM25Okapi([_tokenize(d.page_content) for d in self.docs])

    def retrieve(self, query: str, k: int) -> list[Document]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.docs[i] for i in ranked]


class HybridRetriever:
    """
    Hybrid retrieval: fuses the dense and sparse rankings with RRF.

        RRF(d) = sum_r 1 / (k + rank_r(d))

    where rank_r(d) is the position of d in ranking r and k is a constant.
    """

    def __init__(self):
        self.dense = DenseRetriever()
        self.sparse = BM25Retriever()
        self.rrf_k = CONFIG["retrieval"]["rrf_k"]
        self.n_cand = CONFIG["retrieval"]["candidates_per_branch"]

    def retrieve(self, query: str, k: int) -> list[Document]:
        dense_docs = self.dense.retrieve(query, self.n_cand)
        sparse_docs = self.sparse.retrieve(query, self.n_cand)

        scores: dict[str, float] = {}
        registry: dict[str, Document] = {}
        for ranking in (dense_docs, sparse_docs):
            for rank, doc in enumerate(ranking):
                cid = doc.metadata["chunk_id"]
                registry[cid] = doc
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [registry[cid] for cid, _ in fused]


def get_retriever(kind: str):
    """Retriever factory: 'dense' | 'sparse' | 'hybrid'."""
    if kind == "dense":
        return DenseRetriever()
    if kind == "sparse":
        return BM25Retriever()
    if kind == "hybrid":
        return HybridRetriever()
    raise ValueError(f"Unsupported retriever: {kind}")
