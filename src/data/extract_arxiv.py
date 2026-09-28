"""
Data extraction: download papers from arXiv.

No dataset is provided for this work, so building the corpus is part of it.
This module downloads metadata and PDFs from arXiv according to the query
defined in config/config.yaml and stores an index with the corpus metadata.

Usage:
    python -m src.data.extract_arxiv
"""

from __future__ import annotations

import json
import time

import arxiv
from tqdm import tqdm

from src.config import CONFIG, RAW_DIR


def build_query() -> str:
    """Return the arXiv search query defined in the configuration."""
    return CONFIG["data"]["arxiv_query"]


def download_corpus() -> list[dict]:
    """Download the PDFs and return the list of paper metadata."""
    query = build_query()
    max_papers = CONFIG["data"]["max_papers"]

    print(f"[arXiv] Query: {query}")
    print(f"[arXiv] Downloading up to {max_papers} papers into {RAW_DIR} ...")

    client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    metadata: list[dict] = []
    for result in tqdm(client.results(search), total=max_papers):
        arxiv_id = result.get_short_id().replace("/", "_")
        pdf_path = RAW_DIR / f"{arxiv_id}.pdf"
        # Download the PDF only if it is not already present (idempotent)
        if not pdf_path.exists():
            try:
                result.download_pdf(dirpath=str(RAW_DIR), filename=pdf_path.name)
                time.sleep(1)  # be polite to the arXiv servers
            except Exception as exc:  # noqa: BLE001
                print(f"  ! Error downloading {arxiv_id}: {exc}")
                continue
        metadata.append(
            {
                "arxiv_id": arxiv_id,
                "title": result.title.strip().replace("\n", " "),
                "authors": [a.name for a in result.authors],
                "primary_category": result.primary_category,
                "categories": result.categories,
                "published": result.published.isoformat(),
                "summary": result.summary.strip().replace("\n", " "),
                "pdf_file": pdf_path.name,
            }
        )

    # Store the metadata index
    index_path = RAW_DIR / "corpus_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[arXiv] Downloaded {len(metadata)} papers.")
    print(f"[arXiv] Metadata index: {index_path}")
    return metadata


if __name__ == "__main__":
    download_corpus()
