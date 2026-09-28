"""
Comparative evaluation of the three RAG variants with the RAGAS framework.

Metrics:
  - faithfulness       : answer claims supported by the context (anti-hallucination)
  - answer_relevancy   : relevance of the answer to the question
  - context_precision  : precision of the retrieved context
  - context_recall     : coverage of the reference answer by the context

RAGAS needs a judge LLM and an embedding model; both are set to the providers
configured in .env (by default the same Claude model and local embeddings), so
the evaluation does not depend on a different API.

Outputs:
  - results/eval_<variant>.json : answers and contexts per question (traceability)
  - results/metrics.csv         : table of mean metrics per variant
  - figures/eval_comparison.pdf : bar chart comparing the variants

Usage (requires an API key in .env):
    python -m src.evaluation.evaluate
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.config import CONFIG, FIGURES_DIR, RESULTS_DIR, ROOT

sns.set_theme(style="whitegrid")
PALETTE = {"simple": "#007BC2", "hybrid": "#F05A0F", "agentic": "#822291"}
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load_eval_set() -> list[dict]:
    path = ROOT / CONFIG["evaluation"]["eval_set"]
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_variant(variant_name: str, eval_set: list[dict]):
    """Run one variant over all questions and build the RAGAS dataset."""
    # Imported here so that plotting does not load the whole RAG stack
    from datasets import Dataset

    from src.rag.variants import get_variant

    system = get_variant(variant_name)
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    records = []

    for item in tqdm(eval_set, desc=f"Variant {variant_name}"):
        out = system.answer(item["question"])
        rows["question"].append(item["question"])
        rows["answer"].append(out["answer"])
        rows["contexts"].append(out["contexts"])
        rows["ground_truth"].append(item.get("ground_truth", ""))
        records.append({**item, **out})

    # Store the raw answers (traceability)
    with open(RESULTS_DIR / f"eval_{variant_name}.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return Dataset.from_dict(rows)


def evaluate_all() -> pd.DataFrame:
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from src.rag.llm import get_embeddings, get_llm

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    eval_set = load_eval_set()
    variants = CONFIG["evaluation"]["variants"]

    # Judge LLM and embeddings used by RAGAS to compute the metrics
    evaluator_llm = LangchainLLMWrapper(get_llm())
    evaluator_emb = LangchainEmbeddingsWrapper(get_embeddings())

    all_scores = []
    for name in variants:
        print(f"\n=== Evaluating variant: {name} ===")
        ds = run_variant(name, eval_set)
        result = evaluate(
            ds, metrics=metrics, llm=evaluator_llm, embeddings=evaluator_emb
        )
        scores = result.to_pandas()[METRICS].mean()
        scores["variant"] = name
        all_scores.append(scores)
        # Incremental save: an interruption does not lose finished variants
        pd.DataFrame(all_scores).set_index("variant").to_csv(
            RESULTS_DIR / "metrics.csv"
        )
        print(f"[partial save] {name} -> results/metrics.csv")

    df = pd.DataFrame(all_scores).set_index("variant")
    df.to_csv(RESULTS_DIR / "metrics.csv")
    print("\n=== Mean metrics per variant ===")
    print(df.round(3))
    print(f"  -> {RESULTS_DIR / 'metrics.csv'}")
    return df


def plot_comparison(df: pd.DataFrame) -> None:
    """Grouped bar chart of the four metrics for each variant."""
    x = range(len(METRICS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, variant in enumerate(df.index):
        vals = [df.loc[variant, m] for m in METRICS]
        ax.bar(
            [p + i * width for p in x],
            vals,
            width,
            label=variant,
            color=PALETTE.get(variant),
        )
    ax.set_xticks([p + width for p in x])
    ax.set_xticklabels(
        ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision", "Context\nRecall"]
    )
    ax.set_ylabel("Score (0-1)")
    ax.set_ylim(0, 1)
    ax.set_title("Comparison of the three RAG variants (RAGAS)")
    ax.legend(title="Variant")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eval_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "eval_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {FIGURES_DIR / 'eval_comparison.pdf'}")


def plot_from_csv() -> None:
    """Re-draw the comparison chart from results/metrics.csv (no API calls)."""
    df = pd.read_csv(RESULTS_DIR / "metrics.csv", index_col="variant")
    plot_comparison(df)


def main() -> None:
    df = evaluate_all()
    plot_comparison(df)
    print("\n[Evaluation] Done.")


if __name__ == "__main__":
    main()
