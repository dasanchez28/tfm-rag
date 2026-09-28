"""
The three RAG variants compared in this work:

  1. SimpleRAG   - dense retrieval + generation (baseline).
  2. HybridRAG   - hybrid retrieval (dense + BM25, RRF fusion) + generation.
  3. AgenticRAG  - hybrid retrieval + iterative reasoning (LangGraph) with
                   context grading and faithfulness verification.

All variants expose the same ``answer(question) -> dict`` method, so the
evaluation treats them homogeneously.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document

from src.config import CONFIG
from src.rag.generation import format_context, generate_answer
from src.rag.llm import get_llm
from src.rag.retrievers import get_retriever

TOP_K = CONFIG["retrieval"]["top_k"]


def _result(answer: str, docs: list[Document], iterations: int) -> dict:
    """Common output format of the three variants."""
    return {
        "answer": answer,
        "contexts": [d.page_content for d in docs],
        "sources": [d.metadata.get("arxiv_id") for d in docs],
        "iterations": iterations,
    }


# ------------------------------------------------------------------
# 1. Simple RAG (baseline): dense retrieval + generation
# ------------------------------------------------------------------
class SimpleRAG:
    name = "simple"

    def __init__(self):
        self.retriever = get_retriever("dense")

    def answer(self, question: str) -> dict:
        docs = self.retriever.retrieve(question, TOP_K)
        return _result(generate_answer(question, docs), docs, 1)


# ------------------------------------------------------------------
# 2. Hybrid RAG: hybrid retrieval (RRF) + generation
# ------------------------------------------------------------------
class HybridRAG:
    name = "hybrid"

    def __init__(self):
        self.retriever = get_retriever("hybrid")

    def answer(self, question: str) -> dict:
        docs = self.retriever.retrieve(question, TOP_K)
        return _result(generate_answer(question, docs), docs, 1)


# ------------------------------------------------------------------
# 3. Agentic RAG: state graph with context grading and verification
# ------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    query: str
    docs: list[Document]
    answer: str
    iterations: int
    context_ok: bool
    faithful: bool


class AgenticRAG:
    """
    RAG agent built as a state graph (LangGraph):

        retrieve -> grade -> [reformulate -> retrieve | generate]
        generate -> verify -> [reformulate -> retrieve | end]

    The loop is bounded by max_iterations to control cost and latency.
    """

    name = "agentic"

    def __init__(self):
        self.retriever = get_retriever("hybrid")
        self.llm = get_llm()
        self.max_iter = CONFIG["agent"]["max_iterations"]
        self._graph = self._build_graph()

    # --- graph nodes ---
    def _retrieve(self, state: AgentState) -> AgentState:
        state["docs"] = self.retriever.retrieve(state["query"], TOP_K)
        state["iterations"] = state.get("iterations", 0) + 1
        return state

    def _grade_context(self, state: AgentState) -> AgentState:
        """Ask the LLM whether the context is sufficient to answer."""
        prompt = (
            "Assess whether the CONTEXT is sufficient to answer the QUESTION. "
            "Reply only 'YES' or 'NO'.\n\n"
            f"CONTEXT:\n{format_context(state['docs'])}\n\n"
            f"QUESTION: {state['question']}"
        )
        verdict = self.llm.invoke(prompt).content.strip().upper()
        state["context_ok"] = verdict.startswith("YES")
        return state

    def _reformulate(self, state: AgentState) -> AgentState:
        """Rewrite the query to improve the next retrieval."""
        prompt = (
            "Rewrite the following question to improve the retrieval of "
            "relevant technical documents. Return only the new query.\n\n"
            f"Question: {state['question']}"
        )
        state["query"] = self.llm.invoke(prompt).content.strip()
        return state

    def _generate(self, state: AgentState) -> AgentState:
        state["answer"] = generate_answer(state["question"], state["docs"])
        return state

    def _verify(self, state: AgentState) -> AgentState:
        """Check that the answer is supported by the context (anti-hallucination)."""
        prompt = (
            "Check whether the ANSWER is fully supported by the CONTEXT. "
            "Reply only 'YES' or 'NO'.\n\n"
            f"CONTEXT:\n{format_context(state['docs'])}\n\n"
            f"ANSWER: {state['answer']}"
        )
        verdict = self.llm.invoke(prompt).content.strip().upper()
        state["faithful"] = verdict.startswith("YES")
        return state

    # --- conditional edges ---
    def _after_grade(self, state: AgentState) -> str:
        if state["context_ok"] or state["iterations"] >= self.max_iter:
            return "generate"
        return "reformulate"

    def _after_verify(self, state: AgentState) -> str:
        if state["faithful"] or state["iterations"] >= self.max_iter:
            return "end"
        return "reformulate"

    def _build_graph(self):
        from langgraph.graph import END, StateGraph

        g = StateGraph(AgentState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("grade", self._grade_context)
        g.add_node("reformulate", self._reformulate)
        g.add_node("generate", self._generate)
        g.add_node("verify", self._verify)

        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges(
            "grade",
            self._after_grade,
            {"generate": "generate", "reformulate": "reformulate"},
        )
        g.add_edge("reformulate", "retrieve")
        g.add_edge("generate", "verify")
        g.add_conditional_edges(
            "verify",
            self._after_verify,
            {"end": END, "reformulate": "reformulate"},
        )
        return g.compile()

    def answer(self, question: str) -> dict:
        init: AgentState = {
            "question": question,
            "query": question,
            "docs": [],
            "answer": "",
            "iterations": 0,
            "context_ok": False,
            "faithful": False,
        }
        final = self._graph.invoke(init)
        return _result(final["answer"], final["docs"], final["iterations"])


def get_variant(name: str):
    """Variant factory: 'simple' | 'hybrid' | 'agentic'."""
    return {"simple": SimpleRAG, "hybrid": HybridRAG, "agentic": AgenticRAG}[name]()
