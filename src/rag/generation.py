"""
Grounded answer generation, shared by the three RAG variants.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.config import ANSWER_LANGUAGE
from src.rag.llm import get_llm

SYSTEM_PROMPT = (
    "You are an expert assistant for technical documentation. Answer ONLY "
    "with information contained in the provided CONTEXT. If the context does "
    "not contain the answer, say so explicitly instead of making it up. "
    "Cite the relevant sources by their identifier in square brackets, "
    "e.g. [arxiv_id]. Write your answer in {answer_language}."
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "CONTEXT:\n{context}\n\nQUESTION: {question}\n\n"
            "Grounded answer (with [arxiv_id] citations):",
        ),
    ]
)


def format_context(docs: list[Document]) -> str:
    """Serialise the retrieved chunks, each preceded by its source identifier."""
    blocks = []
    for d in docs:
        src = d.metadata.get("arxiv_id", "unknown")
        blocks.append(f"[{src}] {d.page_content}")
    return "\n\n".join(blocks)


def generate_answer(question: str, docs: list[Document]) -> str:
    """Generate an answer conditioned on the retrieved context."""
    chain = ANSWER_PROMPT | get_llm()
    resp = chain.invoke(
        {
            "context": format_context(docs),
            "question": question,
            "answer_language": ANSWER_LANGUAGE,
        }
    )
    return resp.content
