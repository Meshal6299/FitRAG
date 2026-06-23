"""
FitRAG — Streamlit chat UI (bonus deliverable).

A minimal, dark-themed ChatGPT/Claude-style chat over the FitRAG knowledge base.
Ask a fitness question → get a grounded answer + the sources it came from.

Run:  .venv/Scripts/streamlit run app.py
The RAG pipeline is reused verbatim from src/evaluation.py so the UI answers
exactly the way the evaluation harness does (same retriever, prompt, generator).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# --- make src/ importable and reuse the eval harness's pipeline builders -----
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import evaluation as fe  # noqa: E402

# ---------------------------------------------------------------------------
# Page config + dark styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="FitRAG", page_icon="🏋️", layout="centered")

st.markdown(
    """
    <style>
      /* hide Streamlit's default top toolbar / header */
      [data-testid="stHeader"] { display: none; }
      [data-testid="stToolbar"] { display: none; }
      #MainMenu, footer { visibility: hidden; }
      .block-container { max-width: 820px; padding-top: 2.2rem; }
      /* tighten chat bubbles */
      [data-testid="stChatMessage"] { background: transparent; }
      /* source cards */
      .src-card {
        background: #1a1c22; border: 1px solid #2a2d36; border-radius: 10px;
        padding: 0.7rem 0.9rem; margin-bottom: 0.55rem;
      }
      .src-head { color: #10a37f; font-weight: 600; font-size: 0.9rem; }
      .src-snip { color: #b5b8c0; font-size: 0.82rem; margin-top: 0.3rem; line-height: 1.35; }
      .app-title { font-size: 2rem; font-weight: 700; margin-bottom: 0.1rem; }
      .app-sub   { color: #8b8f9c; font-size: 0.92rem; margin-bottom: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Demo system prompt (stricter than the harness baseline: pins the answer to the
# specific population named in the question, e.g. "beginner").
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are FitRAG, a fitness assistant for beginners.
Answer ONLY for the specific person described in the question.
If the question says 'beginner', answer only for beginners.
Do NOT generalise to athletes or advanced populations.
Answer ONLY using the information provided in the context below.
Do NOT use any knowledge from your training data — only the context.
If context is insufficient say exactly:
"I don't have enough information in my knowledge base to answer this question."
Cite the source document and page at the end of your answer.
Context:
{context}
"""


# ---------------------------------------------------------------------------
# Cached pipeline components
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading retriever…")
def get_vectorstore(embedding_model: str, index_path: str):
    embeddings = fe.build_embeddings(embedding_model)
    return fe.load_vectorstore(index_path, embeddings)


@st.cache_resource(show_spinner="Connecting to model…")
def get_generator(backend: str, model: str):
    return fe.build_generator({
        "llm_backend": backend,
        "llm_model": model,
        "temperature": 0,
        "max_tokens": 1024,
    })


@st.cache_resource(show_spinner="Loading judge…")
def get_judge(backend: str, model: str):
    return fe.build_judge({"judge_backend": backend, "judge_model": model})


# ---------------------------------------------------------------------------
# Fixed configuration (baseline pipeline; no sidebar)
# ---------------------------------------------------------------------------
backend, model = "groq", "llama-3.3-70b-versatile"   # generator
k = 5                                                  # sources retrieved
live_eval = True                                       # judge faithfulness after each answer
# Judge is a different model FAMILY from the llama generator (bias-avoidance).
judge_backend, judge_model = "groq", "openai/gpt-oss-120b"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def render_sources(docs):
    """Render retrieved source documents as compact cards."""
    if not docs:
        return
    with st.expander(f"📚 Sources ({len(docs)})", expanded=False):
        for i, d in enumerate(docs, start=1):
            source = os.path.basename(d.metadata.get("source", "Unknown"))
            page = d.metadata.get("page", "?")
            snippet = " ".join(d.page_content.split())[:280]
            st.markdown(
                f"<div class='src-card'>"
                f"<div class='src-head'>{i}. {source} · page {page}</div>"
                f"<div class='src-snip'>{snippet}…</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


def stream_answer(generator, prompt):
    """Yield answer tokens for st.write_stream; tolerant of non-streaming backends."""
    try:
        for chunk in generator.stream(prompt):
            yield chunk.content if hasattr(chunk, "content") else str(chunk)
    except Exception:  # noqa: BLE001 — fall back to a single blocking call
        resp = generator.invoke(prompt)
        yield resp.content if hasattr(resp, "content") else str(resp)


def retrieval_relevance(vectorstore, query: str, k: int):
    """Objective retrieval-confidence signal (no gold label needed): how closely
    the KB's best chunks match the query, as a 0-1 relevance score from FAISS."""
    try:
        scored = vectorstore.similarity_search_with_relevance_scores(query, k=k)
        scores = [max(0.0, min(1.0, s)) for _, s in scored]
        if not scores:
            return None, None
        return max(scores), sum(scores) / len(scores)
    except Exception:  # noqa: BLE001
        return None, None


def compute_live_metrics(vectorstore, query, answer, context, docs, refused):
    """All metrics that are computable for an arbitrary (unlabelled) question.
    Faithfulness uses the LLM-judge; the rest are objective/free."""
    top_rel, mean_rel = retrieval_relevance(vectorstore, query, k)
    metrics = {
        "refused": refused,
        "n_sources": len(docs),
        "answer_words": len(answer.split()),
        "relevance_top": top_rel,
        "relevance_mean": mean_rel,
        "faithfulness": None,
        "faithfulness_reason": "",
    }
    # Faithfulness only meaningful for a grounded, non-refusal answer.
    if live_eval and not refused and context:
        with st.spinner("Evaluating answer…"):
            judge = get_judge(judge_backend, judge_model)
            res = fe.judge_faithfulness(judge, context, answer, sleep=0)
        metrics["faithfulness"] = res.get("faithfulness")
        metrics["faithfulness_reason"] = res.get("faithfulness_reason", "")
    return metrics


def render_metrics(m):
    """Render the live evaluation panel from a stored metrics dict."""
    if not m:
        return
    with st.expander("📊 Answer evaluation", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        faith = m.get("faithfulness")
        c1.metric("Faithfulness", f"{faith:.2f}" if faith is not None else "—",
                  help="LLM-judge: are the answer's claims grounded in the retrieved context? (0–1)")
        c2.metric("Refused", "Yes" if m.get("refused") else "No",
                  help="Did the model decline (out-of-scope safety)?")
        top_rel = m.get("relevance_top")
        c3.metric("Retrieval (top)", f"{top_rel:.2f}" if top_rel is not None else "—",
                  help="Best query↔chunk similarity in the KB (0–1, objective).")
        mean_rel = m.get("relevance_mean")
        c4.metric("Retrieval (mean)", f"{mean_rel:.2f}" if mean_rel is not None else "—",
                  help="Mean similarity of the top-k chunks (0–1).")

        if m.get("faithfulness_reason"):
            st.caption(f"Judge: _{m['faithfulness_reason']}_")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("<div class='app-title'>🏋️ FitRAG</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='app-sub'>Your evidence-grounded fitness assistant. "
    "Ask about training, exercise, or physical activity.</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Chat state + replay
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("docs"))
            render_metrics(msg.get("metrics"))


# ---------------------------------------------------------------------------
# Chat input → retrieve → generate → show answer + sources
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask a fitness question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            vectorstore = get_vectorstore(
                fe.BASELINE_CONFIG["embedding_model"],
                fe.BASELINE_CONFIG["index_path"],
            )
            retriever = fe.build_retriever(vectorstore, {
                "search_type": fe.BASELINE_CONFIG["search_type"],
                "k": k,
                "fetch_k": fe.BASELINE_CONFIG["fetch_k"],
                "lambda_mult": fe.BASELINE_CONFIG["lambda_mult"],
            })
            generator = get_generator(backend, model)

            with st.spinner("Retrieving sources…"):
                docs = retriever.invoke(prompt)
                context = fe.format_context(docs)

            full_prompt = (
                SYSTEM_PROMPT.format(context=context)
                + "\n\n"
                + fe.HUMAN_PROMPT.format(question=prompt)
            )

            answer = st.write_stream(stream_answer(generator, full_prompt))
            refused = fe.is_refusal(answer)

            # Only show sources when the model actually used the context.
            shown_docs = [] if refused else docs
            render_sources(shown_docs)

            metrics = compute_live_metrics(
                vectorstore, prompt, answer, context, docs, refused
            ) if live_eval else None
            render_metrics(metrics)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer,
                 "docs": shown_docs, "metrics": metrics}
            )
        except Exception as e:  # noqa: BLE001 — surface errors in-UI
            err = f"⚠️ Something went wrong: {e}"
            st.error(err)
            st.session_state.messages.append(
                {"role": "assistant", "content": err, "docs": [], "metrics": None}
            )
