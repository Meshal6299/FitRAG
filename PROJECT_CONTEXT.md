# FitRAG — Complete Project Context

This document gives a full briefing on the FitRAG university project. Use it to ask for help with the report, viva preparation, code questions, or further analysis.

---

## What This Project Is

**FitRAG** is a domain-specific Retrieval-Augmented Generation (RAG) system that answers fitness and exercise-science questions for beginners. It retrieves relevant passages from a curated corpus of 10 PDF documents, then uses a large language model to generate grounded, source-cited answers.

**Course:** CSAI-413 Natural Language Processing Applications, British University in Dubai
**Team:** Meshal Alrajaby, Essam Alsobeh, Ahmed Abdulaziz, Mahmoud Mahmoud
**Grading:** D1 = 30% (submitted, scored 78/100) | D2 = 50% (experiments + evaluation) | Viva = 20%
**Professor feedback on D1:** "Conclusions should be supported by systematic testing and quantitative evidence rather than qualitative observations alone."

---

## Knowledge Base (10 PDFs in `data/raw/`)

| File | Content |
|---|---|
| WHO.pdf | WHO Global Physical Activity Guidelines |
| NSCA_1.pdf | NSCA Position Statement — Resistance Training |
| NSCA_2.pdf | NSCA Position Statement — Youth Resistance Training |
| NSCA_3.pdf | NSCA Position Statement — Older Adults |
| NSCA_4.pdf | NSCA Position Statement — LTAD (Long-Term Athlete Development) |
| NSCA_5.pdf | NSCA Position Statement — Collegiate Strength & Conditioning |
| SSW.pdf | Starting Strength — Weightlifting fundamentals |
| ProgressiveOverload...pdf | Research paper on progressive overload |
| Endurance...pdf | Research paper on endurance training |
| (10th PDF) | Additional fitness reference |

**Key corpus challenge:** Five NSCA documents share dense overlapping vocabulary (resistance training, safety, youth, periodisation). Dense embedding search struggles to discriminate between them — this causes vocabulary-collision retrieval failures.

---

## System Architecture (Baseline Pipeline)

```
User query
    ↓
[Embedding] multi-qa-MiniLM-L6-cos-v1 (HuggingFace, 384-dim, CPU, L2-normalised)
    ↓
[FAISS Vector Store] 2997 chunks, chunk_size=512, overlap=50, separators=["\n\n","\n","."," "]
    ↓
[Retriever] MMR, k=5, fetch_k=20, lambda_mult=0.5
    ↓
[Generator] llama-3.3-70b-versatile via Groq API (or llama3.1:8b via Ollama locally)
    ↓
[Strict System Prompt] Answer ONLY from context. Refuse with fixed sentence if context insufficient.
    ↓
Answer with source citations
```

**Preprocessing:** PyPDF loader → clean (strip page numbers, excessive whitespace) → RecursiveCharacterTextSplitter → drop chunks < 100 chars → embed → FAISS index.

**Refusal sentence (exact):** "I don't have enough information in my knowledge base to answer this question."

---

## Evaluation Framework

### Gold Set (`data/eval/qa_set.json`)
30 questions manually constructed:
- **20 in-scope** — fitness/exercise questions answerable from the PDFs; each has `expected_sources` (which PDF) and `reference_answer`
- **5 ambiguous** — borderline questions that may or may not be answerable
- **5 out-of-scope** — questions outside the fitness domain (nutrition supplements, sleep, psychology) — the system must REFUSE these

### 7 Metrics

| Metric | Type | Description |
|---|---|---|
| Recall@k | Retrieval (objective) | Fraction of expected source PDFs present in top-k retrieved chunks |
| Precision@k | Retrieval (objective) | Fraction of retrieved chunks matching an expected source |
| MRR | Retrieval (objective) | Mean Reciprocal Rank of first expected-source hit |
| Hit-rate | Retrieval (objective) | Whether ≥1 expected source was retrieved |
| Correctness | LLM-as-judge | Factual match vs reference answer, 1–5 scale normalised 0–1 |
| Faithfulness | LLM-as-judge | All answer claims grounded in retrieved context, 1–5 normalised |
| Refusal accuracy | Robustness | Out-of-scope → refuse; in-scope → answer |

### Evaluation Harness
Single function: `evaluate(config) → {aggregate, per_question}` in `src/fitrag_eval.py`.
Every experiment is one call with one changed parameter. Results saved to `data/eval/results/<name>.json`.

### Judge Setup
- **Primary judge:** `qwen2.5:7b-instruct` via Ollama (local, unlimited, `format="json"`)
- **Why a different family:** judge is Qwen (different family from Llama generator) → reduces self-evaluation bias
- **Validation:** compared against Groq `gpt-oss-120b` on already-scored answers → correctness within-1 agreement ≈88%, Pearson r≈0.64; faithfulness within-1 ≈77%
- **Temperature=0** for both generator and judge → fully deterministic, reproducible

---

## Baseline Results

**Config:** MMR k=5, chunk_size=512, multi-qa-MiniLM, strict prompt, Groq llama-3.3-70b generator, gpt-oss-120b judge
**File:** `data/eval/results/baseline.json`

| Metric | Value |
|---|---|
| Recall@5 (in_scope) | 0.925 |
| Recall@5 (all) | 0.868 |
| Precision@5 | 0.575 |
| MRR | 0.771 |
| Hit-rate | 0.917 |
| Correctness (in_scope) | 0.588 |
| Correctness (ambiguous) | 0.500 |
| Faithfulness | 0.978 |
| Refusal accuracy | 0.960 |
| Out-of-scope refused | 1.000 |
| In-scope refused | 0.050 |

**Key signal:** Retrieval is strong; answer correctness (0.588) is the weak link. The 70B model has high faithfulness (0.978) — generator size matters.

**Local baseline** (llama3.1:8b generator + qwen judge, same retrieval, `h2_cs512.json`):
Correctness=0.637, Faithfulness=0.833, Refusal accuracy=0.920

---

## Experiment H1 — Retrieval Strategy: MMR vs Similarity

**Hypothesis:** MMR outperforms similarity on Recall@5 (rejected in D1 qualitative analysis).
**Verdict: REJECTED**

| Config | Recall@5 (in_scope) | Precision@5 | MRR | Hit-rate |
|---|---|---|---|---|
| Similarity k=5 | 0.925 | **0.642** | **0.778** | 0.917 |
| **MMR k=5 (baseline)** | 0.925 | 0.575 | 0.771 | 0.917 |
| MMR k=10 | 0.925 | 0.562 | 0.776 | 0.958 |
| MMR λ=0.8 | 0.925 | 0.650 | 0.785 | 0.917 |
| MMR λ=0.2 | 0.875 | 0.533 | 0.748 | 0.875 |

**Answer quality (separate groq-judged runs):** Correctness tied at 0.625 for both MMR and similarity at k=5.

**Why rejected:** In-scope Recall@5 identical (0.925). Similarity is better on Precision and MRR because MMR's diversity penalty trades relevance for source variety. λ-sweep confirms the mechanism (λ→1 means MMR→similarity). MMR's real benefit is source diversity, not recall accuracy.

---

## Experiment H2 — Chunk Size: 256 vs 512 vs 1024

**Hypothesis:** Smaller chunks (256) improve retrieval precision.
**Verdict: REJECTED**

Index sizes: 6,394 (256) / 2,997 (512) / 1,514 (1024). Overlap fixed at 50. Same retriever (MMR k=5).
All answer-quality runs: local llama3.1:8b gen + qwen judge (consistent across sizes).

| chunk_size | Recall@5 (in_scope) | Precision@5 | MRR | Hit-rate | Correctness | Faithfulness |
|---|---|---|---|---|---|---|
| 256 | 0.825 | 0.575 | 0.689 | 0.833 | 0.662 | 0.716 |
| **512 (baseline)** | **0.925** | 0.575 | 0.771 | **0.917** | 0.637 | 0.833 |
| 1024 | 0.875 | **0.608** | **0.792** | 0.917 | 0.588 | **0.838** |

**Why rejected:** 256 is the worst on recall, MRR, and hit-rate — fragmentation splits relevant content across too many shallow chunks. 512 maximises recall. 1024 maximises precision, MRR, and faithfulness. **Faithfulness rises monotonically** (0.716→0.833→0.838): larger chunks = more complete context = generator needs to supplement less from training data.

---

## Experiment H3 — Embedding Model

**Hypothesis:** `all-MiniLM-L6-v2` or `bge-small-en-v1.5` achieves higher Recall@5 than `multi-qa-MiniLM`.
**Verdict: REJECTED**

Same chunk_size=512, same retriever. Answer quality: local llama gen + qwen judge.

| Embedding model | Recall@5 (in_scope) | Precision@5 | MRR | Hit-rate | Correctness | Faithfulness | Refusal acc. |
|---|---|---|---|---|---|---|---|
| **multi-qa-MiniLM (baseline)** | **0.925** | 0.575 | 0.771 | **0.917** | 0.637 | 0.833 | 0.920 |
| all-MiniLM-L6-v2 | 0.875 | **0.650** | **0.781** | 0.875 | 0.625 | 0.787 | 0.880 |
| bge-small-en-v1.5 | 0.875 | 0.617 | 0.750 | 0.833 | **0.675** | **0.893** | **0.960** |

**Why rejected:** `multi-qa-MiniLM` was fine-tuned on 215M QA pairs — QA domain-match outweighs model size. `all-MiniLM` optimises general semantic similarity. **Counter-intuitive finding:** `bge-small` has worst retrieval recall but best answer correctness (0.675) and faithfulness (0.893) — its fewer but higher-precision hits are sufficient for the generator. This shows retrieval precision matters as much as recall for answer quality.

---

## Experiment H4 — Prompt Design: Strict vs Lenient

**Hypothesis:** Strict-grounded prompt produces higher faithfulness than lenient prompt.
**Verdict: CONFIRMED**

Same retrieval (identical index + retriever). Only prompt changes. Local llama gen + qwen judge.

| Prompt | Correctness (in_scope) | Faithfulness | Refusal acc. | Out-of-scope refused | In-scope refused |
|---|---|---|---|---|---|
| **Strict (baseline)** | 0.637 | **0.833** | **0.920** | **1.000** | 0.050 |
| Lenient | **0.688** | 0.717 | 0.800 | **0.000** | 0.000 |

**Key findings:**
- Faithfulness drops −11.6 pp with lenient prompt (generator uses training-data knowledge)
- **Critical safety failure:** Lenient prompt answered ALL 5 out-of-scope questions (refused rate 0.000) — completely breaks the domain boundary
- Lenient correctness advantage (+5.1 pp) is artefactual — model draws on pretraining, not the knowledge base
- Strict prompt is the correct and necessary design choice

---

## Comparative Analysis — RAG vs Direct LLM

**Setup:** `use_retrieval=False` — same generator, no retrieval, no context injected. Natural ablation to test whether the knowledge base adds value.

| System | Correctness (in_scope) | Faithfulness | Refusal acc. | Out-of-scope refused |
|---|---|---|---|---|
| **RAG (baseline, strict)** | 0.637 | **0.833** | **0.920** | **1.000** |
| Direct LLM (no retrieval) | **0.713** | N/A | 0.800 | **0.000** |

**Key findings:**
- Direct LLM scores higher raw correctness (+7.6 pp) — llama3.1:8b carries fitness knowledge from pretraining and answers freely
- Direct LLM answered ALL 5 out-of-scope questions — completely unscoped
- Faithfulness unmeasurable for direct LLM (no retrieved context to ground against)
- RAG's correctness penalty is partly the strict prompt cost (H4 showed −5.1 pp) and partly 8B model size
- **RAG provides three things direct LLM cannot:** grounding (verifiable, citable), domain safety (refusal), updatability (swap PDFs without retraining)

---

## Failure Analysis (4 Types)

### 1. Retrieval Failure — Q06
**Question:** "Is resistance training safe for children and adolescents according to the NSCA youth position statement?"
**Expected:** NSCA_2.pdf | **Retrieved:** NSCA_5.pdf ×4, NSCA_4.pdf ×1 | **Recall@5 = 0.000**
**Root cause:** Vocabulary collision — all 5 NSCA docs share dense shared vocabulary; embedding cannot distinguish document identity
**Outcome:** Silent failure — 70B model answered correctly (correctness=1.0) from NSCA_5 content, but cited the wrong document
**Fix:** Larger k (k=10 raises Hit-rate to 0.958 overall); principled fix = hybrid BM25+dense retrieval (keyword "youth position statement" discriminates)

### 2. Hallucination — Q11
**Question:** "Does the WHO recommend any physical activity even for people who cannot meet the full guidelines?"
**Retrieved:** All 5 chunks from WHO.pdf — perfect retrieval (Recall@5=1.0)
**Faithfulness = 0.5** — judge: "One claim not supported by the provided context"
**Root cause:** 70B model adds a factually correct claim from pretraining that is absent from the specific retrieved chunks
**Outcome:** Correctness=1.0 but faithfulness=0.5 — answer is accurate but unverifiable from context
**Fix:** Strict prompt already applied (H4: faithfulness 0.833 strict vs 0.717 lenient); residual fix = post-generation claim-level faithfulness checker (bonus feature)

### 3. Ambiguous Query — Q22
**Question:** "Is more training always better?" (6 words)
**Expected:** NSCA_4.pdf + SSW.pdf | **Retrieved:** NSCA_2, ProgressiveOverload.pdf, WHO.pdf, NSCA_5 ×2 | **Recall@5=0.000**
**Root cause:** Short open-ended query embeds in a region of semantic space shared by many fitness documents; intended sources (overtraining/recovery) are not closest cosine neighbours
**Outcome:** Silent failure — 70B answered correctly (correctness=1.0) from tangentially related context
**Fix:** Query expansion before embedding (HyDE / step-back prompting); or flag queries < 8 words for user clarification

### 4. Irrelevant Context — Q20
**Question:** "What recovery considerations does the NSCA LTAD position statement emphasize?"
**Expected:** NSCA_4.pdf | **Retrieved:** NSCA_2, NSCA_1, NSCA_3, **NSCA_4** (rank 4), NSCA_1 | **Recall@5=1.0 but WRONG REFUSAL**
**Root cause:** Correct source at rank 4 surrounded by 4 irrelevant NSCA chunks. Retrieved NSCA_4 chunk covers a different section (not recovery). Generator sees noisy context and triggers refusal.
**Outcome:** False negative — system refuses a question it has the answer for; standard retrieval metrics reported this as a success (Hit=1.0) masking the chunk-level failure
**Fix:** Larger chunk size (H2: MRR rises to 0.792 at 1024, relevant chunks rank higher); higher MMR λ (H1: Precision 0.650 at λ=0.8 reduces noise chunks)

---

## Consolidated Results Table (All 15 Configs)

| Experiment | Configuration | Recall@5 | Prec@5 | MRR | Hit | Correct. | Faith. | Refusal |
|---|---|---|---|---|---|---|---|---|
| **Baseline** | MMR k=5, cs512, multi-qa, strict | 0.925 | 0.575 | 0.771 | 0.917 | 0.637 | 0.833 | 0.920 |
| H1 | Similarity k=5 | 0.925 | 0.642 | 0.778 | 0.917 | — | — | — |
| H1 | MMR k=3 | 0.825 | 0.583 | 0.750 | 0.833 | — | — | — |
| H1 | MMR k=10 | 0.925 | 0.562 | 0.776 | 0.958 | — | — | — |
| H1 | MMR λ=0.8 | 0.925 | 0.650 | 0.785 | 0.917 | — | — | — |
| H2 | chunk=256 | 0.825 | 0.575 | 0.689 | 0.833 | 0.662 | 0.716 | 0.960 |
| H2 | chunk=512 (baseline) | 0.925 | 0.575 | 0.771 | 0.917 | 0.637 | 0.833 | 0.920 |
| H2 | chunk=1024 | 0.875 | 0.608 | 0.792 | 0.917 | 0.588 | 0.838 | 0.880 |
| H3 | multi-qa-MiniLM (baseline) | 0.925 | 0.575 | 0.771 | 0.917 | 0.637 | 0.833 | 0.920 |
| H3 | all-MiniLM-L6-v2 | 0.875 | 0.650 | 0.781 | 0.875 | 0.625 | 0.787 | 0.880 |
| H3 | bge-small-en-v1.5 | 0.875 | 0.617 | 0.750 | 0.833 | 0.675 | 0.893 | 0.960 |
| H4 | Strict (baseline) | 0.925 | 0.575 | 0.771 | 0.917 | 0.637 | 0.833 | 0.920 |
| H4 | Lenient | 0.925 | 0.575 | 0.771 | 0.917 | 0.688 | 0.717 | 0.800 |
| RAG vs LLM | RAG baseline | 0.925 | 0.575 | 0.771 | 0.917 | 0.637 | 0.833 | 0.920 |
| RAG vs LLM | Direct LLM | 0.000 | 0.000 | 0.000 | 0.000 | 0.713 | N/A | 0.800 |

*Answer-quality columns use local qwen2.5:7b judge throughout for cross-experiment comparability. H1 AQ rows left blank (judged in separate groq-judged run, not mixing judge populations).*

---

## Key Findings & Cross-Experiment Insights

1. **The baseline is well-chosen.** 3 of 4 hypotheses rejected (H1, H2, H3). Every D1 design decision survives systematic testing. The experiments retroactively justify the baseline rather than replacing it.

2. **The strict prompt is the most impactful design decision** (H4 confirmed). It enforces faithfulness (+11.6 pp vs lenient) AND maintains safety (out-of-scope refused 1.000 vs 0.000). Non-negotiable for a domain-scoped assistant.

3. **Generator size is the largest lever on quality.** 70B (Groq baseline): Faithfulness=0.978. 8B (local): Faithfulness=0.833. A 14.5 pp gap that no retrieval or chunking configuration in this experiment matrix can close.

4. **Correctness and faithfulness are in tension.** The lenient prompt, direct LLM, and bge-small all score higher on correctness by drawing on pretraining knowledge — but at the cost of faithfulness or refusal safety. For FitRAG, faithfulness and domain safety are the binding constraints.

5. **Silent retrieval failures are harder than metrics suggest.** Q06 and Q22 both show correct answers from wrong sources. Retrieval metrics (Recall@k, Hit-rate) flag these as failures, but the user experience was fine. Real-world failure rates are lower than metrics suggest — but source citations are unreliable in these cases.

6. **Five similar NSCA documents are the primary retrieval challenge.** Three of four failure cases involve incorrect discrimination between NSCA PDFs. Hybrid BM25+dense retrieval would address this specifically.

---

## Report Structure (D2 — Complete)

| Section | Status | Key content |
|---|---|---|
| 1. Introduction | Done | Project overview, D1 feedback, D2 approach |
| 2. System Improvements | Done | 4 axes investigated, single-axis experiment design |
| 3. Evaluation Framework | Done | 7 metrics, gold set, harness, LLM-as-judge setup |
| 4. Hypothesis-Driven Experiments | Done | H1 rejected, H2 rejected, H3 rejected, H4 confirmed |
| 5. Controlled Experiment Summary | Done | Master table (15 configs), cross-experiment insights |
| 6. Failure Analysis | Done | 4 types with example, explanation, attempted fix |
| 7. Comparative Analysis | Done | RAG vs direct LLM — refusal safety is the decisive difference |
| 8. Results and Insights | Done | 6 cross-experiment findings |
| 9. Conclusion | Done | Baseline justified, strict prompt essential, generator size the gap |
| AI Usage Disclosure | Done | Claude, ChatGPT, Gemini used for code + report drafting; all decisions by team |

**Code snippet locations in report:**
1. Section 2 — Strict vs lenient prompt templates (shows exact H4 change)
2. Section 3 — `BASELINE_CONFIG` dict + one `evaluate()` call (shows single-axis design)
3. Section 3 — `_CORRECTNESS_PROMPT` (shows how LLM-as-judge works)
4. Section 4/H2 — `build_index(chunk_size, ...)` call (single-axis isolation)
5. Section 5 — Consolidated results table code (all 15 configs from result files)
6. Section 6 — `show_failures()` function (automated, systematic failure detection)

---

## Viva Preparation — Key Questions & Answers

**Q: Why RAG over direct LLM?**
A: Grounding (every claim traces to a specific retrieved chunk, verifiable), domain safety (refusal when context is insufficient — direct LLM answered all 5 out-of-scope questions), and updatability (swap PDFs without retraining). Direct LLM scored higher raw correctness (0.713 vs 0.637) but answered everything — it is unscoped.

**Q: What is Recall@k?**
A: The fraction of expected source documents present in the top-k retrieved chunks. If a question expects NSCA_2.pdf and WHO.pdf, and both appear in the top 5 retrieved chunks, Recall@5 = 1.0.

**Q: Why was MMR kept if it lost to similarity on every metric?**
A: MMR's benefit is source diversity, not recall accuracy. It prevents all 5 chunks coming from the same document, which ensures the generator sees multiple perspectives. We retain it while acknowledging the D1 claim ("MMR improves recall") was wrong.

**Q: Why did 256-token chunks perform worst?**
A: Fragmentation. Splitting text into very short chunks breaks passages mid-sentence and mid-idea. The relevant content for a question is spread across dozens of tiny chunks that individually score too low to surface in the top 5. 512 is the sweet spot for this corpus.

**Q: Why is faithfulness important? Isn't correctness enough?**
A: A model can be correct AND unfaithful — Q11 shows this (Correctness=1.0, Faithfulness=0.5). The model added a factually true claim that wasn't in the retrieved context. For a fitness assistant giving health advice, an unverifiable claim is a liability even if it happens to be correct. Faithfulness is the RAG system's promise: every claim is traceable to a specific source chunk.

**Q: What is the biggest remaining weakness?**
A: Generator size. The 8B local model produces Faithfulness=0.833; the 70B Groq model produces 0.978. No retrieval or chunking change in our experiments closes that 14.5 pp gap. A larger generator or a domain-fine-tuned model is the highest-leverage next step.

**Q: What is a silent retrieval failure?**
A: A case where standard retrieval metrics (Recall@k, Hit-rate) report failure, but the generated answer is correct. Q06 and Q22 both received correct answers from wrong source documents. The user experience is fine, but the source citations are wrong — which undermines the core value proposition of a grounded, citable assistant.

**Q: Why use an LLM as judge instead of ROUGE or exact match?**
A: ROUGE measures lexical overlap, not semantic correctness. A paraphrased correct answer would score near zero on ROUGE but 1.0 on correctness. The brief mandates measuring answer correctness and hallucination presence — these are semantic judgements that require understanding. LLM-as-judge at temperature=0 is reproducible and reference-anchored.

**Q: Why is the qwen judge different enough from the llama generator to avoid self-evaluation bias?**
A: They are different model families (Qwen vs LLaMA), trained on different data with different RLHF. The bias we avoid is a model rating its own outputs more leniently than an independent judge would. Using a different family breaks the correlation between generator and judge preferences.

---

## File Structure

```
FitRAG/
├── src/
│   ├── fitrag_eval.py          # Full evaluation harness — evaluate(config) entry point
│   ├── build_index.py          # Rebuild FAISS index at any chunk_size / embedding model
│   └── NLPA_Group_Project_2026.pdf  # Original project brief
├── notebooks/
│   ├── 01_preprocess.ipynb     # Load PDFs, clean, chunk
│   ├── 02_embed.ipynb          # Embed + build baseline FAISS index
│   ├── 03_rag_pipeline.ipynb   # End-to-end RAG demo
│   └── 04_evaluation.ipynb     # All experiments, results, failure analysis (main D2 notebook)
├── embeddings/
│   ├── vector_store/           # Baseline (multi-qa-MiniLM, cs512)
│   ├── vector_store_cs256/     # H2 chunk=256
│   ├── vector_store_cs512/     # H2 chunk=512 (same as baseline)
│   ├── vector_store_cs1024/    # H2 chunk=1024
│   ├── vector_store_emb_allMiniLM/   # H3 all-MiniLM
│   └── vector_store_emb_bge/         # H3 bge-small
├── data/
│   ├── raw/                    # 10 source PDFs
│   └── eval/
│       ├── qa_set.json         # 30-question gold set
│       └── results/            # All experiment result files
│           ├── baseline.json           # 70B Groq gen + gpt-oss judge
│           ├── h1_retrieval_sweep.json # H1 retrieval-only sweep
│           ├── h2_chunksize_sweep.json # H2 retrieval-only sweep
│           ├── h2_cs256.json           # H2 256 AQ (local gen + qwen judge)
│           ├── h2_cs512.json           # H2 512 AQ = local pipeline baseline reference
│           ├── h2_cs1024.json          # H2 1024 AQ
│           ├── h3_embedding_sweep.json # H3 retrieval-only sweep
│           ├── h3_allMiniLM.json       # H3 all-MiniLM AQ
│           ├── h3_bge.json             # H3 bge-small AQ
│           ├── h4_lenient.json         # H4 lenient prompt AQ
│           ├── direct_llm.json         # RAG vs direct LLM
│           ├── local_baseline.json     # H1 AQ — MMR (groq judge)
│           └── sim_k5_local.json       # H1 AQ — similarity (groq judge)
└── .env                        # GROQ_API_KEY (gitignored)
```

---

## Team Roles (for Report Assembly)

| Member | Report sections | Code snippets |
|---|---|---|
| Meshal Alrajaby | Introduction, Evaluation Framework | Snippet 2 (BASELINE_CONFIG + evaluate call), Snippet 3 (judge prompts) |
| Essam Alsobeh | H1, H2 experiments | Snippet 4 (build_index call) |
| Ahmed Abdulaziz | H3, H4 experiments, Failure Analysis | Snippet 1 (strict vs lenient prompts), Snippet 6 (show_failures) |
| Mahmoud Mahmoud | Comparative Analysis, Results & Insights, Conclusion, AI Disclosure, PDF assembly | Snippet 5 (consolidated results table) |
