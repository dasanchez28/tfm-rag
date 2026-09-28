"""
Web application: chat interface (Streamlit).

Features:
  - Selection of the RAG variant (simple / hybrid / agentic).
  - Chat with a conversation history kept during the session.
  - Sources panel listing the arXiv papers that support each answer.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make the src package importable when running through streamlit
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.variants import get_variant  # noqa: E402

VARIANTS = ["simple", "hybrid", "agentic"]
VARIANT_LABELS = {
    "simple": "Simple RAG (baseline)",
    "hybrid": "Hybrid RAG (RRF)",
    "agentic": "Agentic RAG (LangGraph)",
}

st.set_page_config(page_title="RAG Agent · Technical Documentation", page_icon="📄")

# --- Session state ---
if "history" not in st.session_state:
    st.session_state.history = []
if "variant_name" not in st.session_state:
    st.session_state.variant_name = "agentic"
if "system" not in st.session_state:
    st.session_state.system = None


@st.cache_resource(show_spinner="Loading the RAG system ...")
def load_system(name: str):
    return get_variant(name)


def show_sources(sources: list[str]) -> None:
    """Render the unique sources, in order, with a link to arXiv."""
    with st.expander("📚 Sources"):
        for s in dict.fromkeys(sources):
            st.markdown(f"- `{s}`  ·  https://arxiv.org/abs/{s}")


# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    variant = st.selectbox(
        "System variant",
        options=VARIANTS,
        index=VARIANTS.index(st.session_state.variant_name),
        format_func=VARIANT_LABELS.get,
    )
    if variant != st.session_state.variant_name:
        st.session_state.variant_name = variant
        st.session_state.system = None
    if st.button("🗑️ Clear conversation"):
        st.session_state.history = []
    st.markdown("---")
    st.caption("Master's Thesis · MSc in Artificial Intelligence (09MIAR) · VIU")

st.title("📄 RAG Agent for Technical Documentation")
st.caption(f"Active variant: **{st.session_state.variant_name}**")

if st.session_state.system is None:
    st.session_state.system = load_system(st.session_state.variant_name)

# --- Conversation history ---
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            show_sources(turn["sources"])

# --- User input ---
if prompt := st.chat_input("Ask a question about the technical corpus..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            out = st.session_state.system.answer(prompt)
        st.markdown(out["answer"])
        show_sources(out["sources"])
        if out.get("iterations", 1) > 1:
            st.caption(f"🔁 The agent iterated {out['iterations']} times.")

    st.session_state.history.append(
        {"role": "assistant", "content": out["answer"], "sources": out["sources"]}
    )
