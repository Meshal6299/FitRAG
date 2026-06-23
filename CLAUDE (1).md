# CLAUDE.md

Guidance + **living progress tracker** for this repository.

> **Maintenance rule:** This file tracks project progress. Whenever a step is finished, update the
> relevant checklist below (`[ ]` → `[x]`), bump the **Status** line, and add a dated note under
> **Progress log**. Keep it honest — only check items that actually exist and work in the repo.

---

## Status

- **Current phase:** Final stretch — Demo + D2 Report + Viva prep
- **D1:** ✅ Submitted, **scored 78/100**
- **D2:** 🚧 All experiments done. **3 things left: Streamlit app, D2 report PDF, AI Usage Disclosure.**
- **Last updated:** 2026-06-23

**Baseline numbers (30-Q gold set, `data/eval/results/baseline.json`):** Recall@k(in_scope)=0.925,
Hit-rate=0.917, MRR=0.771, Precision@k=0.575, Correctness(in_scope)=0.588, Faithfulness=0.978,
Refusal-accuracy=0.96. Key signal: retrieval is strong but **answer correctness is the weak link** —
the judge marks many in_scope answers partial/incomplete.

**Pipeline score: 8.5/10.** Retrieval excellent, safety perfect, faithfulness strong.
Weak points: answer correctness (0.588), NSCA vocabulary collision, minimal text cleaning.

### Professor feedback on D1 (drives D2 priorities)
> "For Deliverable 2, focus on strengthening the experimental analysis. In particular, compare
> retrieval strategies, chunking configurations, and evaluation metrics using **quantitative
> evidence rather than qualitative observations alone**. The discussion of MMR and retrieval
> failures is a good starting point, but stronger conclusions should be supported by **systematic
> testing**."

---

## ✅ What is done

- [x] Gold evaluation set — `data/eval/qa_set.json` (30 Qs: 20 in-scope / 5 ambiguous / 5 out-of-scope)
- [x] Evaluation harness — `src/fitrag_eval.py`, `evaluate(config) → metrics`, 7 metrics implemented
- [x] H1: MMR vs similarity — **REJECTED** (similarity ≥ MMR on all retrieval metrics at k=5)
- [x] H2: Chunk size 256/512/1024 — **REJECTED** (512 best recall; 256 worst; faithfulness monotonic with size)
- [x] H3: Embedding model comparison — **REJECTED** (multi-qa-MiniLM already best on recall)
- [x] H4: Strict vs lenient prompt — **CONFIRMED** (strict faithfulness 0.833 vs lenient 0.717; lenient breaks refusal safety)
- [x] RAG vs direct LLM — done (direct LLM correctness 0.713 but refused 0/5 out-of-scope)
- [x] Failure analysis — 4 types documented (Q06 retrieval, Q11 hallucination, Q22 ambiguous, Q20 irrelevant context)
- [x] Consolidated results table — 15 configs in `04_evaluation.ipynb`

---

## 🔴 What is left — in priority order

### Step 1 — Fix the demo prompt (30 minutes)
**Problem:** System returned "hybrid athlete" for a beginner query — the prompt does not
constrain the population being addressed.

**Fix:** Add one line to `SYSTEM_PROMPT` in `notebooks/03_rag_pipeline.ipynb` and `app.py`:

```python
SYSTEM_PROMPT = """You are FitRAG, a fitness assistant for beginners.
Answer ONLY for the specific person described in the question.
If the question says 'beginner', answer only for beginners.
Do NOT generalise to athletes or advanced populations.
Answer ONLY using the information provided in the context below.
Do NOT use any knowledge from your training data — only the context.
If the context does not contain enough information to answer, say exactly:
"I don't have enough information in my knowledge base to answer this question."
When possible, mention which source your answer comes from.
Keep answers clear and beginner-friendly.

Context:
{context}
"""
```

**After fixing, test these 5 queries and screenshot every output:**
1. "How many days per week should a beginner train?" → should answer with 2-3 days, cite SSW/WHO
2. "What is progressive overload?" → should give clear definition, cite progressive_overload.pdf
3. "How many sets and reps should a beginner do?" → should say 2-3 sets, 10-15 reps, cite NSCA_3
4. "What is the best workout?" → should REFUSE (ambiguous)
5. "What should I eat before training?" → should REFUSE (out-of-scope)

- [ ] Prompt fixed in `03_rag_pipeline.ipynb`
- [ ] All 5 test queries produce correct outputs
- [ ] Screenshots taken for report Section 5.2

---

### Step 2 — Build the Streamlit app (2-3 hours)
**File:** `app.py` (currently empty)
**Purpose:** Bonus marks + live demo for viva

**Minimum viable app — paste this into `app.py`:**

```python
import streamlit as st
from dotenv import load_dotenv
import os, sys, json

load_dotenv(".env")
sys.path.insert(0, "src")

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Config ────────────────────────────────────────────────────────────────
INDEX_PATH   = "embeddings/vector_store"
MODEL_NAME   = "multi-qa-MiniLM-L6-cos-v1"
GROQ_MODEL   = "llama-3.3-70b-versatile"

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

# ── Load pipeline (cached so it only loads once) ──────────────────────────
@st.cache_resource
def load_pipeline():
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.load_local(
        INDEX_PATH, embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
    )
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=1024,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Question: {question}")
    ])
    def fmt(docs):
        return "\n\n".join(
            f"[Source {i+1}: {os.path.basename(d.metadata.get('source','?'))}, "
            f"page {d.metadata.get('page','?')}]\n{d.page_content}"
            for i, d in enumerate(docs)
        )
    chain = (
        {"context": retriever | fmt, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return retriever, chain

# ── UI ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FitRAG", page_icon="🏋️", layout="centered")
st.title("🏋️ FitRAG — Fitness Assistant")
st.caption("Answers grounded in sports science research and WHO guidelines")
st.info("Ask any fitness question. Answers are grounded in 10 curated documents.", icon="💡")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

query = st.chat_input("Ask a fitness question...")

if query:
    st.chat_message("user").write(query)
    retriever, chain = load_pipeline()

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            docs = retriever.invoke(query)
            answer = chain.invoke(query)
        st.write(answer)

        with st.expander(f"📂 Sources ({len(docs)} chunks retrieved)"):
            for i, doc in enumerate(docs):
                src  = os.path.basename(doc.metadata.get("source", "Unknown"))
                page = doc.metadata.get("page", "?")
                st.markdown(f"**{i+1}. {src} · page {page}**")
                st.caption(doc.page_content[:300] + "...")

    st.session_state.messages.append({"role": "user",      "content": query})
    st.session_state.messages.append({"role": "assistant", "content": answer})
```

**Run with:** `streamlit run app.py` from the project root (not from notebooks/).

**Screenshots to take for report:**
- App running with a good answer + sources expanded
- App showing a refusal on an out-of-scope query
- App showing sources panel with chunk previews

- [ ] `app.py` written and runs without errors
- [ ] Good answer screenshot taken
- [ ] Refusal screenshot taken
- [ ] Sources panel screenshot taken

---

### Step 3 — Write the D2 report PDF (4-5 hours)
**All content is already in `PROJECT_CONTEXT.md` and `04_evaluation.ipynb`.**
This is a copy-and-format job, not a research job.

**Write sections in this order (fastest path):**

#### 3.1 Title page
Same format as D1. Update date to submission date.

#### 3.2 AI Usage Disclosure ← do this FIRST, professor flagged it in D1
Fill in the table below. Every member signs.

| Tool | How it was used | What was done independently |
|------|----------------|----------------------------|
| Claude (Anthropic) | Report drafting assistance, code explanation, project planning | All design decisions, experiment interpretation, conclusions |
| ChatGPT | Brainstorming experiment hypotheses | All analysis and write-up |
| Groq API (llama-3.3-70b) | LLM generator in RAG pipeline | Pipeline architecture, evaluation framework design |
| Ollama (llama3.1:8b, qwen2.5:7b) | Local generator and judge | Experimental setup, result interpretation |

| Name | Student ID | Signature |
|------|-----------|-----------|
| Meshal Alrajaby | 22000101 | _________ |
| Essam Al Sobeh | 22000882 | _________ |
| Ahmed Abdulaziz | 22000427 | _________ |
| Mahmoud Mahoud | 22000796 | _________ |

#### 3.3 Section 1 — Introduction (half page)
Three paragraphs:
1. What FitRAG is + what D1 delivered (2 sentences)
2. Quote professor feedback verbatim
3. What D2 does in response: 30-Q gold set, 7-metric harness, 4 controlled experiments,
   LLM-as-judge validated at temperature=0

#### 3.4 Section 2 — System Improvements (1 page)
One paragraph per axis. For each: what you tested and why. Keep brief — numbers come later.
- Retrieval strategy (MMR vs similarity) — tested because D1 qualitatively claimed MMR was better
- Chunk size (256/512/1024) — tested because D1 identified 512 as possibly too large
- Embedding model (3 models) — tested to empirically justify D1's choice
- Prompt design (strict vs lenient) — tested because D1 had no quantitative hallucination measure

#### 3.5 Section 3 — Evaluation Framework (1.5 pages)
Key things to cover:
- The 30-Q gold set: 20 in-scope (with reference answers + expected sources),
  5 ambiguous, 5 out-of-scope. Why each category exists.
- All 7 metrics in a table (copy from PROJECT_CONTEXT.md)
- LLM-as-judge: why (ROUGE is semantics-blind), how (temp=0, qwen family ≠ llama family),
  validation (88% within-1 agreement vs gpt-oss-120b)
- Code snippet 1: `BASELINE_CONFIG` dict + one `evaluate()` call

#### 3.6 Section 4 — Hypothesis-Driven Experiments (3-4 pages, biggest section)
**Each hypothesis: hypothesis → setup → results table → interpretation**

**H1 (MMR vs similarity):**
- Hypothesis: MMR outperforms similarity on Recall@5
- Setup: swept k∈{3,5,10} and λ∈{0.2,0.5,0.8} vs similarity k=5; held chunk/embedding constant
- Results table: copy H1 table from PROJECT_CONTEXT.md
- Interpretation: REJECTED — tied on recall, similarity better on precision/MRR;
  λ-sweep confirms diversity penalty mechanism; D1 qualitative claim was wrong

**H2 (chunk size):**
- Hypothesis: smaller chunks (256) improve retrieval precision
- Setup: rebuilt index at 256/512/1024, overlap fixed at 50, same retriever
- Results table: copy H2 table from PROJECT_CONTEXT.md
- Interpretation: REJECTED — 256 worst on all metrics; 512 best recall;
  faithfulness monotonic with chunk size (0.716→0.833→0.838)

**H3 (embedding model):**
- Hypothesis: all-MiniLM or bge-small achieves higher Recall@5 than multi-qa-MiniLM
- Setup: rebuilt index with each model, same chunk/retrieval
- Results table: copy H3 table from PROJECT_CONTEXT.md
- Interpretation: REJECTED — multi-qa-MiniLM already best; QA fine-tuning outweighs model size;
  counter-intuitive bge-small finding (worst recall, best correctness)

**H4 (prompt design):**
- Hypothesis: strict prompt produces higher faithfulness than lenient prompt
- Setup: identical retrieval, only prompt changes; local gen + qwen judge
- Results table: copy H4 table from PROJECT_CONTEXT.md
- Interpretation: CONFIRMED — faithfulness 0.833 vs 0.717 (−11.6 pp);
  critical safety finding: lenient answered all 5 out-of-scope (refused 0.000 vs 1.000)

**Code snippet 2:** strict vs lenient prompt templates side by side

#### 3.7 Section 5 — Controlled Experiment Summary (1 page)
Paste the 15-config consolidated table from PROJECT_CONTEXT.md.
Write 6 bullet points — one per cross-experiment insight from PROJECT_CONTEXT.md.

#### 3.8 Section 6 — Failure Analysis (1.5 pages)
Four subsections. For each: query → retrieved sources → system output → root cause → fix → result.
- 6.1 Q06: Retrieval failure — NSCA vocabulary collision
- 6.2 Q11: Hallucination — perfect retrieval but 70B adds ungrounded claim
- 6.3 Q22: Ambiguous query — 6-word query embeds too broadly
- 6.4 Q20: Irrelevant context — correct source at rank 4, drowned by noise

**Code snippet 3:** `show_failures()` function

#### 3.9 Section 7 — Comparative Analysis (1 page)
RAG vs direct LLM. Use the table from PROJECT_CONTEXT.md.
Key narrative: direct LLM wins on correctness (+7.6 pp) but answers ALL out-of-scope questions.
RAG's value = grounding + domain safety + updatability. Correctness gap = strict prompt cost.

#### 3.10 Section 8 — Results and Insights (1 page)
6 numbered findings from PROJECT_CONTEXT.md. Each one as a short paragraph with numbers.

#### 3.11 Section 9 — Conclusion (half page)
Three paragraphs:
1. Baseline empirically justified — 3/4 hypotheses rejected, D1 design choices survived
2. Strict prompt is the most impactful single decision (H4 confirmed, safety-critical)
3. Generator size is the largest remaining lever — 70B faithfulness 0.978 vs 8B 0.909;
   no retrieval experiment closes that gap

#### 3.12 References
Cite: LangChain, FAISS, sentence-transformers, HuggingFace, Groq, Ollama,
WHO guidelines, all NSCA position statements, Starting Strength article,
progressive overload papers.

#### 3.13 Appendix
- Full 30-Q gold set (from `data/eval/qa_set.json`)
- Code snippets 1-3 (listed above)
- Screenshots from notebooks and Streamlit app

**Report checklist:**
- [ ] Title page done
- [ ] AI Usage Disclosure filled + signed by all 4 members
- [ ] Section 1 — Introduction
- [ ] Section 2 — System Improvements
- [ ] Section 3 — Evaluation Framework
- [ ] Section 4 — H1 written with table
- [ ] Section 4 — H2 written with table
- [ ] Section 4 — H3 written with table
- [ ] Section 4 — H4 written with table
- [ ] Section 5 — Consolidated 15-config table
- [ ] Section 6 — All 4 failure cases
- [ ] Section 7 — RAG vs direct LLM table + narrative
- [ ] Section 8 — 6 findings
- [ ] Section 9 — Conclusion
- [ ] References
- [ ] Appendix (gold set + code snippets + screenshots)
- [ ] Exported as PDF

---

### Step 4 — Code cleanup and README update (1 hour)

- [ ] All 4 notebooks run clean top-to-bottom without errors
- [ ] `src/fitrag_eval.py` has docstrings on the `evaluate()` function
- [ ] `src/build_index.py` has docstrings
- [ ] README.md updated with D2 section:
  - How to run the Streamlit app (`streamlit run app.py`)
  - How to run the evaluation harness
  - How to run experiments (one `evaluate()` call with different config)
  - List of all result files in `data/eval/results/`
- [ ] All result JSON files committed to repo
- [ ] `.env.example` added (shows keys needed without real values)
- [ ] `app.py` committed and working

---

### Step 5 — Viva preparation (day before Week 10)
**Every member must answer all 9 questions below independently — no notes.**
Run a teach-back session: each person explains the full pipeline to the rest of the team.

**The 9 questions that will come up:**

**Q1: Why RAG over direct LLM?**
Direct LLM scored higher raw correctness (0.713 vs 0.637) but answered ALL 5 out-of-scope
questions — completely unscoped. RAG provides grounding (every claim traceable to a chunk),
domain safety (refusal when context is insufficient), and updatability (swap PDFs without retraining).

**Q2: What is Recall@5?**
The fraction of expected source documents present in the top-5 retrieved chunks.
If a question expects NSCA_2 and WHO, and both appear in top 5 → Recall@5 = 1.0.
Our baseline: 0.925 on in-scope questions.

**Q3: You rejected MMR in H1 but kept it in your baseline — why?**
MMR's benefit is source diversity, not recall accuracy. At k=5 it ties or loses on recall
metrics vs similarity, but it prevents all 5 chunks coming from the same document.
We retain it for diversity while correcting the D1 qualitative claim that it improved recall.

**Q4: Why did 256-character chunks perform worst?**
Fragmentation. Short chunks split passages mid-idea. Relevant content for a question
spreads across dozens of tiny chunks that each score too low to surface in top 5.
512 is the sweet spot — H2 confirmed this empirically with Recall@5 = 0.825 (256) vs 0.925 (512).

**Q5: Why is faithfulness important if correctness is higher?**
Q11 shows this: Correctness=1.0, Faithfulness=0.5. The model added a factually true claim
not present in any retrieved chunk. For a health-adjacent assistant, an unverifiable claim
is a liability even if it happens to be correct. Faithfulness = every claim traceable to a source.

**Q6: What is a silent retrieval failure?**
Retrieval metrics report failure (wrong source retrieved) but the generated answer is correct.
Q06 and Q22: correct answers from wrong source documents. User experience is fine,
but source citations are wrong — undermines the core value of a grounded, citable assistant.

**Q7: Why use LLM-as-judge instead of ROUGE?**
ROUGE measures lexical overlap, not semantic correctness. A paraphrased correct answer
scores near zero on ROUGE. The brief mandates measuring answer correctness and hallucination —
semantic judgements requiring understanding. LLM-as-judge at temperature=0 is reproducible
and reference-anchored. We validated it: 88% within-1 agreement with gpt-oss-120b.

**Q8: Why is qwen the judge if llama is the generator?**
Bias avoidance — a model rates its own outputs more leniently. Using a different model
family (Qwen vs LLaMA, trained separately with different RLHF) breaks that correlation.
Family independence matters, not just provider independence.

**Q9: What is the biggest remaining weakness and what would you do with more time?**
Three things: (1) Hybrid BM25+dense retrieval to solve the NSCA vocabulary collision
(Q06, Q20 failures); (2) A larger generator — 70B faithfulness 0.978 vs 8B 0.909,
that 14.5 pp gap is the largest remaining quality lever; (3) Post-generation claim-level
faithfulness checker as a safety layer for Q11-type hallucinations.

- [ ] All 4 members can answer Q1–Q9 without notes
- [ ] Teach-back session completed
- [ ] Live demo tested end-to-end (app.py runs, 5 test queries all work correctly)

---

## Final submission checklist

- [ ] `app.py` working (Streamlit demo)
- [ ] D2 report PDF exported
- [ ] AI Usage Disclosure in report — all 4 names, IDs, signatures
- [ ] All 4 notebooks run clean
- [ ] README updated with D2 instructions
- [ ] All result JSONs committed
- [ ] CLAUDE.md updated (tick boxes above)
- [ ] Repo pushed to GitHub before deadline

---

## Pipeline configuration (unchanged baseline)

- **Chunking:** `chunk_size=512`, `chunk_overlap=50`, separators `["\n\n","\n","."," "]`; drop <100 chars → **2997 chunks**
- **Embeddings:** `multi-qa-MiniLM-L6-cos-v1` (HF, 384-dim), CPU, normalized
- **Vector store:** FAISS (L2 over normalized vectors ≈ cosine)
- **Retriever:** MMR — `k=5`, `fetch_k=20`, `lambda_mult=0.5`
- **LLM (generator):** `groq` → `llama-3.3-70b-versatile` (default) or `ollama` → `llama3.1:8b` (local, unlimited)
- **Generation prompt:** strict grounded — answer only from context, fixed refusal sentence, cite sources
- **Eval harness:** `src/fitrag_eval.py`, `evaluate(config) → {aggregate, per_question}`; results to `data/eval/results/`
- **LLM-as-judge:** `ollama` → `qwen2.5:7b-instruct` (local, unlimited, validated)

---

## Progress log

- **2026-06-19** — D1 returned, 78/100. Built eval harness, ran all 4 hypotheses, failure analysis, RAG vs LLM. All experiments complete. Only report + app remain.
- **2026-06-23** — CLAUDE.md updated with final step-by-step instructions. Pipeline assessed 8.5/10. 3 items left: demo prompt fix, Streamlit app, D2 report PDF.
