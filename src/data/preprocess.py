"""
Corpus preprocessing: PDF -> clean text -> chunks.

Usage:
    python -m src.data.preprocess
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.config import CONFIG, PROCESSED_DIR, RAW_DIR


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract the text of a PDF, page by page, with PyMuPDF."""
    text_parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts)


def clean_text(text: str) -> str:
    """Basic cleaning: normalise whitespace and remove common artefacts."""
    # Join words hyphenated at the end of a line
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Collapse line breaks and repeated spaces
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove arXiv identifiers stamped on the pages
    text = re.sub(r"arXiv:\d{4}\.\d{4,5}(v\d+)?", " ", text)
    return text.strip()


def build_splitter() -> RecursiveCharacterTextSplitter:
    """Recursive splitter that respects paragraph and sentence boundaries."""
    ck = CONFIG["chunking"]
    return RecursiveCharacterTextSplitter(
        chunk_size=ck["chunk_size"],
        chunk_overlap=ck["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def preprocess_corpus() -> list[dict]:
    """Process the whole corpus and store the chunks with their metadata."""
    index_path = RAW_DIR / "corpus_index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            "corpus_index.json not found. Run src.data.extract_arxiv first."
        )

    with open(index_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    splitter = build_splitter()
    all_chunks: list[dict] = []
    docs_meta: list[dict] = []

    for paper in tqdm(corpus, desc="Processing PDFs"):
        pdf_path = RAW_DIR / paper["pdf_file"]
        if not pdf_path.exists():
            continue
        try:
            raw = extract_text_from_pdf(pdf_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! Error reading {pdf_path.name}: {exc}")
            continue

        cleaned = clean_text(raw)
        chunks = splitter.split_text(cleaned)

        docs_meta.append(
            {
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "primary_category": paper["primary_category"],
                "published": paper["published"],
                "n_chars": len(cleaned),
                "n_chunks": len(chunks),
            }
        )

        for i, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "chunk_id": f"{paper['arxiv_id']}_{i}",
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper["title"],
                    "source": paper["pdf_file"],
                    "text": chunk,
                }
            )

    # Persist the results
    chunks_path = PROCESSED_DIR / "chunks.json"
    docs_path = PROCESSED_DIR / "docs_meta.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(docs_meta, f, ensure_ascii=False, indent=2)

    print(f"Processed documents: {len(docs_meta)}")
    print(f"Generated chunks:    {len(all_chunks)}")
    print(f"  -> {chunks_path}")
    print(f"  -> {docs_path}")
    return all_chunks


if __name__ == "__main__":
    preprocess_corpus()
