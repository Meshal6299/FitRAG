"""
FitRAG evaluation harness (Deliverable 2).

Single entry point: ``evaluate(config) -> metrics``. Every D2 experiment is one
call to ``evaluate`` with a different ``config`` dict, so results stay comparable
and reproducible (professor feedback on D1: quantitative evidence, systematic
testing).

Metrics produced per run
------------------------
Retrieval (objective, uses ``expected_sources`` from the gold set):
    Recall@k, Precision@k, MRR, Hit-rate.
Answer quality (LLM-as-judge — a different model *family* from the generator
(gpt-oss judging llama) to reduce self-evaluation bias, while staying on Groq's
generous rate limits; Gemini's free tier caps at 20 requests/day):
    Correctness (vs ``reference_answer``), Faithfulness (claims grounded in
    retrieved context).
Robustness:
    Refusal accuracy (out_of_scope -> refuse; in_scope -> don't refuse).

The module resolves all paths relative to the repository root (the parent of
this file's ``src/`` directory), so it works whether imported from a notebook in
``notebooks/`` or from the repo root.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
GOLD_SET_PATH = REPO_ROOT / "data" / "eval" / "qa_set.json"
RESULTS_DIR = REPO_ROOT / "data" / "eval" / "results"

load_dotenv(REPO_ROOT / ".env")


# ---------------------------------------------------------------------------
# Default (baseline) configuration — mirrors the current pipeline exactly
# ---------------------------------------------------------------------------
BASELINE_CONFIG: Dict[str, Any] = {
    "name": "baseline",
    # --- retrieval ---
    "embedding_model": "multi-qa-MiniLM-L6-cos-v1",
    "index_path": "embeddings/vector_store",  # relative to REPO_ROOT
    "search_type": "mmr",                      # "mmr" | "similarity"
    "k": 5,
    "fetch_k": 20,
    "lambda_mult": 0.5,
    "use_retrieval": True,                      # False -> direct LLM (no context)
    # --- generation ---
    "llm_backend": "groq",                      # "groq" (API) | "ollama" (local, unlimited)
    "llm_model": "llama-3.3-70b-versatile",
    "temperature": 0,
    "max_tokens": 1024,
    "prompt_variant": "strict",                # "strict" | "lenient"
    # --- judging ---
    "judge": True,
    "judge_backend": "groq",                    # "groq" (API) | "ollama" (local, unlimited)
    "judge_model": "openai/gpt-oss-120b",       # different family from generator -> less self-eval bias
    "judge_sleep": 0.3,                          # seconds between judge calls (rate limit)
}

# Convenience preset for running the generator locally (no API limits). Spread it
# into a config, e.g.  evaluate({"name": "local_baseline", **LOCAL_GEN})
# Judging still runs on Groq gpt-oss-120b (separate quota, never the bottleneck).
LOCAL_GEN: Dict[str, Any] = {
    "llm_backend": "ollama",
    "llm_model": "llama3.1:8b",   # fits the RTX 3080 (10GB) on-GPU; swap for any pulled Ollama model
}

# Convenience preset for running the JUDGE locally (no API limits). qwen2.5:7b is a
# strong, different-family judge (vs the llama generator). Validated against
# gpt-oss-120b via an agreement study. Spread alongside LOCAL_GEN for a fully
# offline, unlimited run:  evaluate({"name": ..., **LOCAL_GEN, **LOCAL_JUDGE})
LOCAL_JUDGE: Dict[str, Any] = {
    "judge_backend": "ollama",
    "judge_model": "qwen2.5:7b-instruct",
    "judge_sleep": 0,             # no rate limit locally
}

# The exact refusal sentence the strict prompt is instructed to emit. We match
# loosely (substring, apostrophe-insensitive) so paraphrases still count.
REFUSAL_SENTENCE = "I don't have enough information in my knowledge base to answer this question."
_REFUSAL_MARKERS = [
    "don't have enough information",
    "do not have enough information",
    "dont have enough information",
]


# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------
STRICT_SYSTEM_PROMPT = """You are FitRAG, an expert fitness assistant for beginners.
You answer questions about exercise, training programs, and physical fitness.

STRICT RULES:
1. Answer ONLY using the information provided in the context below.
2. Do NOT use any knowledge from your training data — only the context.
3. If the context does not contain enough information to answer, say exactly:
"I don't have enough information in my knowledge base to answer this question."
4. When possible, mention which source your answer comes from.
5. Keep answers clear and beginner-friendly.
6. Do not make up specific numbers, studies, or recommendations not in the context.
7. Dont mention the source except at the end

Context:
{context}
"""

LENIENT_SYSTEM_PROMPT = """You are FitRAG, a helpful fitness assistant for beginners.
Use the context below to answer the question. The context is your primary source,
but you may lightly draw on general fitness knowledge to give a clear, complete,
beginner-friendly answer. Mention the source at the end when you used the context.

Context:
{context}
"""

# Used only when use_retrieval is False (RAG vs direct-LLM experiment).
DIRECT_SYSTEM_PROMPT = """You are FitRAG, an expert fitness assistant for beginners.
Answer the question about exercise, training, or physical fitness clearly and for a
beginner audience."""

HUMAN_PROMPT = "Question: {question}"


# ---------------------------------------------------------------------------
# Gold set
# ---------------------------------------------------------------------------
def load_gold_set(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = Path(path) if path else GOLD_SET_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------
def build_embeddings(model_name: str):
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vectorstore(index_path: str, embeddings):
    from langchain_community.vectorstores import FAISS

    abs_path = index_path
    if not os.path.isabs(index_path):
        abs_path = str(REPO_ROOT / index_path)
    return FAISS.load_local(
        abs_path, embeddings, allow_dangerous_deserialization=True
    )


def build_retriever(vectorstore, config: Dict[str, Any]):
    if config["search_type"] == "mmr":
        search_kwargs = {
            "k": config["k"],
            "fetch_k": config["fetch_k"],
            "lambda_mult": config["lambda_mult"],
        }
    else:
        search_kwargs = {"k": config["k"]}
    return vectorstore.as_retriever(
        search_type=config["search_type"], search_kwargs=search_kwargs
    )


def build_generator(config: Dict[str, Any]):
    """Generator backend is selectable via config['llm_backend']:
      - 'groq'   -> ChatGroq (API; subject to free-tier daily token caps)
      - 'ollama' -> ChatOllama (local; unlimited, runs on the GPU)
    The judge stays on Groq regardless (see build_judge)."""
    backend = config.get("llm_backend", "groq")

    if backend == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config["llm_model"],
            temperature=config["temperature"],
            num_predict=config["max_tokens"],          # Ollama's name for max output tokens
            num_ctx=config.get("num_ctx", 8192),        # context window (prompt + 5 chunks fits easily)
            base_url=config.get("ollama_base_url", "http://127.0.0.1:11434"),
        )

    from langchain_groq import ChatGroq

    return ChatGroq(
        model=config["llm_model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


def build_judge(config: Dict[str, Any]):
    """Judge backend is selectable via config['judge_backend']:
      - 'groq'   -> ChatGroq, default gpt-oss-120b (strong, but daily token cap)
      - 'ollama' -> ChatOllama, default qwen2.5:7b-instruct (local, unlimited)

    Either way the judge is a *different model family* from the llama generator,
    which is what reduces self-evaluation bias — not the provider. The local
    judge uses Ollama's format='json' so a small model reliably emits parseable
    JSON (no chain-of-thought to strip). See the agreement study justifying the
    7B local judge vs gpt-oss-120b."""
    backend = config.get("judge_backend", "groq")

    if backend == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config["judge_model"],
            temperature=0,
            num_predict=512,
            num_ctx=8192,
            format="json",   # constrain output to valid JSON — crucial for small judges
            base_url=config.get("ollama_base_url", "http://127.0.0.1:11434"),
        )

    from langchain_groq import ChatGroq

    kwargs = dict(
        model=config["judge_model"],
        temperature=0,
        max_tokens=1024,  # gpt-oss emits chain-of-thought before JSON; leave room for both
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )
    # gpt-oss reasoning models: keep CoT short so the JSON always lands within budget.
    if "gpt-oss" in config["judge_model"]:
        kwargs["reasoning_effort"] = "low"
    return ChatGroq(**kwargs)


def get_system_prompt(config: Dict[str, Any]) -> str:
    if not config.get("use_retrieval", True):
        return DIRECT_SYSTEM_PROMPT
    if config.get("prompt_variant", "strict") == "lenient":
        return LENIENT_SYSTEM_PROMPT
    return STRICT_SYSTEM_PROMPT


def format_context(docs) -> str:
    parts = []
    for i, doc in enumerate(docs):
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source {i + 1}: {source}, page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call helpers (with retry/backoff for rate limits)
# ---------------------------------------------------------------------------
def _invoke_with_retry(llm, prompt, retries: int = 4, base_delay: float = 4.0) -> str:
    """Invoke an LLM, retrying on rate-limit / transient errors with backoff."""
    last_err = None
    for attempt in range(retries):
        try:
            resp = llm.invoke(prompt)
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:  # noqa: BLE001 - we want to retry broadly on API errors
            last_err = e
            msg = str(e).lower()
            transient = any(
                t in msg
                for t in ("429", "resource_exhausted", "rate", "timeout", "503", "overloaded")
            )
            if not transient or attempt == retries - 1:
                break
            time.sleep(base_delay * (2 ** attempt))
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")


def _extract_json(text: str) -> Dict[str, Any]:
    """Pull a {...} object out of a model response and parse it.

    Reasoning models often emit chain-of-thought (which may itself contain
    braces) before the answer, so we scan for the last balanced object that
    parses, falling back to a greedy match.
    """
    # Try each "{" as a start, longest-first, and return the first that parses.
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for start in reversed(starts):
        for end in range(len(text), start, -1):
            if text[end - 1] != "}":
                continue
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                continue
    return {}


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------
def retrieval_metrics(retrieved_sources: List[str], expected_sources: List[str]) -> Dict[str, float]:
    """retrieved_sources is in rank order; expected_sources is the gold set."""
    expected = set(expected_sources)
    if not expected:
        return {}  # not applicable (e.g. out_of_scope)

    retrieved_unique = set(retrieved_sources)
    k = len(retrieved_sources) or 1

    hit = 1.0 if (retrieved_unique & expected) else 0.0
    recall = len(retrieved_unique & expected) / len(expected)
    precision = sum(1 for s in retrieved_sources if s in expected) / k

    rr = 0.0
    for rank, src in enumerate(retrieved_sources, start=1):
        if src in expected:
            rr = 1.0 / rank
            break

    return {
        "hit": hit,
        "recall_at_k": recall,
        "precision_at_k": precision,
        "reciprocal_rank": rr,
    }


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------
def is_refusal(answer: str) -> bool:
    norm = answer.lower().replace("’", "'")
    return any(marker in norm for marker in _REFUSAL_MARKERS)


# ---------------------------------------------------------------------------
# LLM-as-judge metrics
# ---------------------------------------------------------------------------
_CORRECTNESS_PROMPT = """You are a strict grader for a fitness question-answering system.
Compare the SYSTEM ANSWER to the REFERENCE ANSWER for the given QUESTION.

Score factual correctness and completeness on a 1-5 scale:
5 = fully correct and complete, matches the reference.
4 = correct, minor omission.
3 = partially correct, missing or vague on key points.
2 = mostly incorrect or misleading.
1 = wrong, contradicts the reference, or non-answer.

QUESTION: {question}
REFERENCE ANSWER: {reference}
SYSTEM ANSWER: {answer}

Respond with ONLY a JSON object: {{"score": <1-5>, "reason": "<one short sentence>"}}"""

_FAITHFULNESS_PROMPT = """You are checking whether an ANSWER is grounded in the provided CONTEXT.
A faithful answer makes only claims that are supported by the context (no invented
numbers, studies, or facts).

Score on a 1-5 scale:
5 = every claim is supported by the context.
3 = mostly supported, one unsupported or embellished claim.
1 = contains claims clearly not supported by the context (hallucination).

CONTEXT:
{context}

ANSWER: {answer}

Respond with ONLY a JSON object: {{"score": <1-5>, "reason": "<one short sentence>"}}"""


def judge_correctness(judge, question: str, reference: str, answer: str, sleep: float) -> Dict[str, Any]:
    prompt = _CORRECTNESS_PROMPT.format(question=question, reference=reference, answer=answer)
    raw = _invoke_with_retry(judge, prompt)
    time.sleep(sleep)
    parsed = _extract_json(raw)
    score = parsed.get("score")
    norm = (float(score) - 1) / 4 if isinstance(score, (int, float)) else None
    return {"correctness_raw": score, "correctness": norm, "correctness_reason": parsed.get("reason", "")}


def judge_faithfulness(judge, context: str, answer: str, sleep: float) -> Dict[str, Any]:
    prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    raw = _invoke_with_retry(judge, prompt)
    time.sleep(sleep)
    parsed = _extract_json(raw)
    score = parsed.get("score")
    norm = (float(score) - 1) / 4 if isinstance(score, (int, float)) else None
    return {"faithfulness_raw": score, "faithfulness": norm, "faithfulness_reason": parsed.get("reason", "")}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    in_scope = [r for r in records if r["category"] == "in_scope"]
    ambiguous = [r for r in records if r["category"] == "ambiguous"]
    out_scope = [r for r in records if r["category"] == "out_of_scope"]
    with_expected = [r for r in records if r.get("expected_sources")]

    # Refusal accuracy: in_scope should NOT refuse, out_of_scope SHOULD refuse.
    refusal_correct = (
        [1.0 if not r["refused"] else 0.0 for r in in_scope]
        + [1.0 if r["refused"] else 0.0 for r in out_scope]
    )

    return {
        "n_questions": len(records),
        # Retrieval (over all items with expected_sources)
        "recall_at_k": _mean([r.get("recall_at_k") for r in with_expected]),
        "precision_at_k": _mean([r.get("precision_at_k") for r in with_expected]),
        "mrr": _mean([r.get("reciprocal_rank") for r in with_expected]),
        "hit_rate": _mean([r.get("hit") for r in with_expected]),
        # Retrieval restricted to in_scope (cleanest signal)
        "recall_at_k_in_scope": _mean([r.get("recall_at_k") for r in in_scope]),
        # Answer quality
        "correctness_in_scope": _mean([r.get("correctness") for r in in_scope]),
        "correctness_ambiguous": _mean([r.get("correctness") for r in ambiguous]),
        "faithfulness": _mean([r.get("faithfulness") for r in records]),
        # Robustness
        "refusal_accuracy": _mean(refusal_correct),
        "in_scope_refused_rate": _mean([1.0 if r["refused"] else 0.0 for r in in_scope]),
        "out_of_scope_refused_rate": _mean([1.0 if r["refused"] else 0.0 for r in out_scope]),
        "ambiguous_refused_rate": _mean([1.0 if r["refused"] else 0.0 for r in ambiguous]),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def evaluate(
    config: Optional[Dict[str, Any]] = None,
    gold_set: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = True,
    save: bool = True,
    precomputed_answers: Optional[Dict[str, str]] = None,
    retrieval_only: bool = False,
) -> Dict[str, Any]:
    """Run one full evaluation pass over the gold set for a given config.

    Returns a dict with ``config``, ``aggregate`` metrics, and ``per_question``
    records. Optionally writes the result to ``data/eval/results/<name>.json``.

    ``precomputed_answers`` maps question id -> answer text. When supplied, the
    generator is skipped for those ids and the saved answer is reused. Because
    the generator runs at temperature=0, reusing a previous run's answers is
    reproducible — useful for adding judge metrics to an existing run without
    re-spending generation tokens (free-tier daily caps).

    ``retrieval_only`` skips generation and judging entirely and computes only the
    objective retrieval metrics (Recall@k, Precision@k, MRR, Hit-rate). This is
    free and fast — ideal for retrieval sweeps (e.g. H1 MMR vs similarity, k/lambda
    sweeps) where the generator's answer is not under test.
    """
    cfg = {**BASELINE_CONFIG, **(config or {})}
    questions = gold_set if gold_set is not None else load_gold_set()

    if verbose:
        print(f"{'=' * 70}")
        print(f"  EVALUATING CONFIG: {cfg['name']}")
        print(f"  retrieval={cfg['use_retrieval']} | {cfg['search_type']} k={cfg['k']} "
              f"fetch_k={cfg['fetch_k']} lambda={cfg['lambda_mult']}")
        print(f"  embed={cfg['embedding_model']} | prompt={cfg['prompt_variant']}")
        print(f"  gen={cfg['llm_model']} [{cfg.get('llm_backend','groq')}] | judge={cfg['judge_model'] if cfg['judge'] else 'OFF'}")
        print(f"{'=' * 70}")

    # --- build components ---
    embeddings = build_embeddings(cfg["embedding_model"])
    retriever = None
    if cfg["use_retrieval"]:
        vectorstore = load_vectorstore(cfg["index_path"], embeddings)
        retriever = build_retriever(vectorstore, cfg)
    precomputed_answers = precomputed_answers or {}
    need_generator = (not retrieval_only) and any(q["id"] not in precomputed_answers for q in questions)
    generator = build_generator(cfg) if need_generator else None
    judge = build_judge(cfg) if (cfg["judge"] and not retrieval_only) else None

    system_prompt = get_system_prompt(cfg)

    records: List[Dict[str, Any]] = []
    for i, q in enumerate(questions, start=1):
        question = q["question"]
        expected = [os.path.basename(s) for s in q.get("expected_sources", [])]

        # --- retrieve ---
        retrieved_sources: List[str] = []
        context = ""
        if cfg["use_retrieval"]:
            docs = retriever.invoke(question)
            retrieved_sources = [os.path.basename(d.metadata.get("source", "?")) for d in docs]
            context = format_context(docs)

        # --- generate (or reuse a precomputed answer; skipped when retrieval_only) ---
        if retrieval_only:
            answer = None
        elif q["id"] in precomputed_answers:
            answer = precomputed_answers[q["id"]]
        else:
            if cfg["use_retrieval"]:
                full_prompt = system_prompt.format(context=context) + "\n\n" + HUMAN_PROMPT.format(question=question)
            else:
                full_prompt = system_prompt + "\n\n" + HUMAN_PROMPT.format(question=question)
            answer = _invoke_with_retry(generator, full_prompt)

        refused = is_refusal(answer) if answer is not None else False

        rec: Dict[str, Any] = {
            "id": q["id"],
            "category": q["category"],
            "question": question,
            "expected_sources": expected,
            "retrieved_sources": retrieved_sources,
            "answer": answer,
            "refused": refused,
        }
        rec.update(retrieval_metrics(retrieved_sources, expected))

        # --- judge: correctness (skip out_of_scope; those are scored by refusal) ---
        if judge is not None and q["category"] != "out_of_scope":
            rec.update(judge_correctness(judge, question, q["reference_answer"], answer, cfg["judge_sleep"]))

        # --- judge: faithfulness (only meaningful for grounded, non-refusal answers) ---
        if judge is not None and cfg["use_retrieval"] and not refused and context:
            rec.update(judge_faithfulness(judge, context, answer, cfg["judge_sleep"]))

        records.append(rec)

        if verbose:
            flags = []
            if "recall_at_k" in rec:
                flags.append(f"R@k={rec['recall_at_k']:.2f}")
            if rec.get("correctness") is not None:
                flags.append(f"corr={rec['correctness']:.2f}")
            if rec.get("faithfulness") is not None:
                flags.append(f"faith={rec['faithfulness']:.2f}")
            flags.append("REFUSED" if refused else "answered")
            print(f"  [{i:2d}/{len(questions)}] {q['id']} {q['category']:<12} | " + " ".join(flags))

    agg = aggregate(records)
    result = {"config": cfg, "aggregate": agg, "per_question": records}

    if verbose:
        print(f"\n{'-' * 70}\n  AGGREGATE METRICS — {cfg['name']}\n{'-' * 70}")
        for key, val in agg.items():
            if isinstance(val, float):
                print(f"    {key:<28} {val:.3f}")
            else:
                print(f"    {key:<28} {val}")

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"{cfg['name']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"\n  Saved results -> {out_path.relative_to(REPO_ROOT)}")

    return result


if __name__ == "__main__":
    # Quick smoke test: retrieval-only baseline (no judge, fast & free).
    evaluate({"name": "smoke_retrieval_only", "judge": False})
