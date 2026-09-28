# RAG Agent for Technical Documentation

Retrieval-Augmented Generation (RAG) system for reliable question answering over
technical documentation, combining **hybrid retrieval**, **iterative (agentic)
reasoning** and **source faithfulness verification**.

This repository contains the source code of the Master's Thesis *"RAG agent for
technical documentation: hybrid retrieval, iterative reasoning and source
verification"* (MSc in Artificial Intelligence, 09MIAR, Universidad Internacional
de Valencia, 2025-2026).

- **Author:** Diego Aguado Sánchez
- **Supervisor:** Jesús Cigales Canga

## Overview

The system answers questions over a corpus of arXiv papers on RAG, agents and
language models. Three variants of the same RAG system, of increasing
complexity, are implemented and compared. Each variant adds one component to
the previous one, so the comparison works as an ablation study that isolates
the contribution of each component:

| Variant   | Retrieval                         | Reasoning                          | Verification               |
|-----------|-----------------------------------|------------------------------------|----------------------------|
| `simple`  | Dense (vector search)             | None (retrieve-then-generate)      | No                         |
| `hybrid`  | Hybrid (dense + BM25, RRF fusion) | None                               | No                         |
| `agentic` | Hybrid (RRF)                      | Iterative, LangGraph state graph   | Faithfulness to the sources |

The agentic variant retrieves, grades whether the context is sufficient,
reformulates the query when it is not, generates a grounded answer and verifies
that the answer is supported by the context. The loop is bounded by
`max_iterations` to control cost and latency.

## Results

Corpus: 60 arXiv papers, 6,308 chunks (about 175 tokens per chunk).
Evaluation: 30 questions with reference answers, scored with
[RAGAS](https://docs.ragas.io/). Generator and judge model:
`claude-haiku-4-5-20251001`.

| Variant   | Faithfulness | Answer relevancy | Context precision | Context recall |
|-----------|:------------:|:----------------:|:-----------------:|:--------------:|
| `simple`  | 0.836        | 0.600            | 0.447             | 0.391          |
| `hybrid`  | 0.817        | **0.623**        | **0.488**         | 0.508          |
| `agentic` | **0.885**    | 0.540            | 0.434             | **0.530**      |

- Hybrid retrieval mainly improves retrieval quality (context recall +30% over the baseline).
- The agentic variant reaches the highest faithfulness, at the cost of lower
  answer relevancy and more LLM calls per query.

The result files are in `results/` and the figures in `figures/`.

## Repository structure

```
tfm-rag/
├── app/
│   └── streamlit_app.py        # chat interface
├── config/
│   └── config.yaml             # pipeline parameters
├── data/
│   └── eval/
│       └── eval_questions.json # evaluation questions with reference answers
├── figures/                    # EDA and evaluation figures (PDF + PNG)
├── results/                    # eda_summary.json, metrics.csv
├── src/
│   ├── config.py               # configuration loading (.env + YAML)
│   ├── data/
│   │   ├── extract_arxiv.py    # corpus download from arXiv
│   │   └── preprocess.py       # PDF -> text -> chunks
│   ├── eda/
│   │   └── run_eda.py          # exploratory data analysis
│   ├── rag/
│   │   ├── llm.py              # generator LLM and embeddings
│   │   ├── ingest.py           # vector indexing (ChromaDB)
│   │   ├── retrievers.py       # dense, BM25 and hybrid (RRF) retrievers
│   │   ├── generation.py       # grounded answer generation
│   │   └── variants.py         # the three RAG variants
│   └── evaluation/
│       └── evaluate.py         # RAGAS evaluation and comparison chart
├── run_pipeline.py             # pipeline orchestrator
├── requirements.txt
└── .env.example                # environment variables template
```

`data/raw/` (downloaded PDFs) and `data/processed/` (chunks and vector index)
are not versioned; the pipeline regenerates them.

## Installation

Requires Python 3.10 or later.

```bash
git clone https://github.com/<user>/tfm-rag.git
cd tfm-rag
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

Copy the environment template and add your API key:

```bash
cp .env.example .env    # Windows: copy .env.example .env
```

By default the generator is Claude (`LLM_PROVIDER=anthropic`) through the
Anthropic API; OpenAI is also supported. Embeddings run locally with
sentence-transformers, at no cost.

## Usage

Run the pipeline stage by stage:

```bash
python run_pipeline.py --step extract      # 1. download the papers from arXiv
python run_pipeline.py --step preprocess   # 2. PDF -> text -> chunks
python run_pipeline.py --step eda          # 3. exploratory analysis -> figures/
python run_pipeline.py --step ingest       # 4. vector indexing (ChromaDB)
python run_pipeline.py --step evaluate     # 5. RAGAS evaluation (uses the API)
```

or all at once with `python run_pipeline.py --all`.
`python run_pipeline.py --step plot` redraws the evaluation chart from
`results/metrics.csv` without calling the API.

Launch the chat interface (requires the ingest stage):

```bash
streamlit run app/streamlit_app.py
```

## Configuration

All parameters are in `config/config.yaml`:

| Parameter                         | Value | Description                                  |
|-----------------------------------|-------|----------------------------------------------|
| `chunking.chunk_size`             | 800   | Characters per chunk                         |
| `chunking.chunk_overlap`          | 120   | Overlap between consecutive chunks           |
| `retrieval.top_k`                 | 5     | Chunks passed to the generator               |
| `retrieval.rrf_k`                 | 60    | Reciprocal Rank Fusion constant              |
| `retrieval.candidates_per_branch` | 10    | Candidates from each branch before fusion    |
| `agent.max_iterations`            | 3     | Maximum iterations of the agent              |
| `generation.answer_language`      | Spanish | Language of the generated answers          |

## Reproducibility

- All parameters are externalised in `config/config.yaml` and `.env`.
- A fixed random seed (`RANDOM_SEED`) and a low temperature (0.1) are used.
- Models served through an API may change over time: the model version used in
  the reported results is `claude-haiku-4-5-20251001`.
- Only generation and evaluation call the API; the embeddings run locally.
- The code follows PEP 8, is organised in modules and documented with docstrings.

## License

Released under the [MIT License](LICENSE).
