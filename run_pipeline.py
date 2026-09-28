"""
Pipeline orchestrator.

Runs the stages of the technical development in order:
  1. extract    - download the corpus from arXiv
  2. preprocess - PDF -> text -> chunks
  3. eda        - exploratory data analysis (figures + summary)
  4. ingest     - vector indexing in ChromaDB
  5. evaluate   - evaluation of the three variants with RAGAS (requires an API key)

Additional stage:
  plot - re-draw the evaluation chart from results/metrics.csv (no API calls)

Usage:
    python run_pipeline.py --all
    python run_pipeline.py --step extract
    python run_pipeline.py --step plot
"""

from __future__ import annotations

import argparse

from src.config import set_seed

PIPELINE = ["extract", "preprocess", "eda", "ingest", "evaluate"]


def extract():
    from src.data.extract_arxiv import download_corpus

    download_corpus()


def preprocess():
    from src.data.preprocess import preprocess_corpus

    preprocess_corpus()


def eda():
    from src.eda.run_eda import main as run_eda

    run_eda()


def ingest():
    from src.rag.ingest import build_index

    build_index()


def evaluate():
    from src.evaluation.evaluate import main as run_eval

    run_eval()


def plot():
    from src.evaluation.evaluate import plot_from_csv

    plot_from_csv()


STEPS = {
    "extract": extract,
    "preprocess": preprocess,
    "eda": eda,
    "ingest": ingest,
    "evaluate": evaluate,
    "plot": plot,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG pipeline")
    parser.add_argument("--step", choices=list(STEPS), help="Run a single stage.")
    parser.add_argument("--all", action="store_true", help="Run the whole pipeline.")
    args = parser.parse_args()

    set_seed()

    if args.all:
        for name in PIPELINE:
            print(f"\n{'=' * 60}\n  STAGE: {name}\n{'=' * 60}")
            STEPS[name]()
    elif args.step:
        STEPS[args.step]()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
