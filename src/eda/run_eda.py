"""
Exploratory Data Analysis (EDA) of the corpus.

Generates the figures that characterise the corpus:
  1. Temporal distribution of the papers (per year).
  2. Distribution by arXiv category.
  3. Histogram of document length (number of characters).
  4. Histogram of the number of chunks per document.
  5. Distribution of chunk length (number of tokens).
  6. Word cloud and bar chart of the most frequent terms.

Figures are saved in figures/ as PDF (vector) and PNG, and a numeric summary
of the corpus is saved in results/eda_summary.json.

Usage:
    python -m src.eda.run_eda
"""

from __future__ import annotations

import json
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from src.config import FIGURES_DIR, PROCESSED_DIR, RANDOM_SEED, RAW_DIR, RESULTS_DIR

sns.set_theme(style="whitegrid")
ORANGE = "#F05A0F"
BLUE = "#007BC2"
plt.rcParams["figure.dpi"] = 120

# Basic English stop words for the term analysis
STOPWORDS = set(
    """the a an and or of to in for on with is are be as by that this we our it
    from at which can using use used method methods model models results result
    approach based paper propose proposed show shows also more than these those
    such into their they them then thus given per via not no than may""".split()
)

_ENC = None


def _count_tokens(text: str) -> int:
    """Count tokens with tiktoken; fall back to a word-based estimate."""
    global _ENC
    if _ENC is None:
        try:
            import tiktoken

            _ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001  (no network or no tiktoken)
            _ENC = False
    if _ENC:
        return len(_ENC.encode(text))
    # Approximation: ~0.75 words per token in technical English
    return int(len(text.split()) / 0.75)


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> figures/{name}.pdf")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Load the corpus index, the document metadata and the chunks."""
    with open(RAW_DIR / "corpus_index.json", encoding="utf-8") as f:
        corpus = pd.DataFrame(json.load(f))
    with open(PROCESSED_DIR / "docs_meta.json", encoding="utf-8") as f:
        docs = pd.DataFrame(json.load(f))
    with open(PROCESSED_DIR / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    return corpus, docs, chunks


def fig_temporal(corpus: pd.DataFrame) -> None:
    corpus = corpus.copy()
    corpus["year"] = pd.to_datetime(corpus["published"]).dt.year
    counts = corpus["year"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index.astype(str), counts.values, color=ORANGE)
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Number of papers")
    ax.set_title("Temporal distribution of the corpus")
    _save(fig, "eda_temporal")


def fig_categories(corpus: pd.DataFrame) -> None:
    counts = corpus["primary_category"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(counts.index[::-1], counts.values[::-1], color=BLUE)
    ax.set_xlabel("Number of papers")
    ax.set_ylabel("arXiv category (primary)")
    ax.set_title("Distribution by category")
    _save(fig, "eda_categories")


def fig_doc_length(docs: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(docs["n_chars"], bins=20, color=ORANGE, ax=ax)
    ax.set_xlabel("Document length (characters)")
    ax.set_ylabel("Number of documents")
    ax.set_title("Document length distribution")
    _save(fig, "eda_doc_length")


def fig_chunks_per_doc(docs: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(docs["n_chunks"], bins=20, color=BLUE, ax=ax)
    ax.set_xlabel("Chunks per document")
    ax.set_ylabel("Number of documents")
    ax.set_title("Chunks generated per document")
    _save(fig, "eda_chunks_per_doc")


def fig_chunk_tokens(chunks: list[dict]) -> list[int]:
    token_counts = [_count_tokens(c["text"]) for c in chunks]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(token_counts, bins=30, color=ORANGE, ax=ax)
    ax.set_xlabel("Chunk length (tokens)")
    ax.set_ylabel("Number of chunks")
    ax.set_title("Chunk length distribution")
    _save(fig, "eda_chunk_tokens")
    return token_counts


def fig_terms(chunks: list[dict]) -> None:
    text = " ".join(c["text"].lower() for c in chunks)
    words = re.findall(r"[a-z]{3,}", text)
    freq = Counter(w for w in words if w not in STOPWORDS)

    # Word cloud
    wc = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        colormap="autumn",
        max_words=120,
        random_state=RANDOM_SEED,
    ).generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Most frequent terms in the corpus")
    _save(fig, "eda_wordcloud")

    # Top-15 bar chart
    labels, values = zip(*freq.most_common(15))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(labels[::-1], values[::-1], color=BLUE)
    ax.set_xlabel("Frequency")
    ax.set_title("Top 15 most frequent terms")
    _save(fig, "eda_top_terms")


def summary_table(
    corpus: pd.DataFrame,
    docs: pd.DataFrame,
    chunks: list[dict],
    token_counts: list[int],
) -> None:
    """Print and store the numeric summary of the corpus."""
    stats = {
        "n_documents": len(docs),
        "n_chunks": len(chunks),
        "mean_chars_per_doc": round(docs["n_chars"].mean(), 1),
        "mean_chunks_per_doc": round(docs["n_chunks"].mean(), 1),
        "mean_tokens_per_chunk": round(sum(token_counts) / len(token_counts), 1),
        "max_tokens_per_chunk": max(token_counts),
        "n_categories": int(corpus["primary_category"].nunique()),
    }
    out = RESULTS_DIR / "eda_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("\n=== Corpus summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  -> {out}")


def main() -> None:
    print("[EDA] Loading processed data ...")
    corpus, docs, chunks = load_data()
    print("[EDA] Generating figures ...")
    fig_temporal(corpus)
    fig_categories(corpus)
    fig_doc_length(docs)
    fig_chunks_per_doc(docs)
    token_counts = fig_chunk_tokens(chunks)
    fig_terms(chunks)
    summary_table(corpus, docs, chunks, token_counts)
    print("[EDA] Done.")


if __name__ == "__main__":
    main()
