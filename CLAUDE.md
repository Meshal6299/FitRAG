# CLAUDE.md

Guidance + **living progress tracker** for this repository.

> **Maintenance rule:** This file tracks project progress. Whenever a step is finished, update the
> relevant checklist below (`[ ]` → `[x]`), bump the **Status** line, and add a dated note under
> **Progress log**. Keep it honest — only check items that actually exist and work in the repo.

---

## Status

- **Current phase:** Deliverable 2 — Experimental Investigation + Evaluation
- **D1:** ✅ Submitted, **scored 78/100**
- **D2:** 🚧 All experiments + failure analysis + consolidated results table done. **Only D2 report PDF + AI Usage Disclosure remain.**
- **Last updated:** 2026-06-19

**Baseline numbers (30-Q gold set, `data/eval/results/baseline.json`):** Recall@k(in_scope)=0.925,
Hit-rate=0.917, MRR=0.771, Precision@k=0.575, Correctness(in_scope)=0.588, Faithfulness=0.978,
Refusal-accuracy=0.96. Key signal: retrieval is strong but **answer correctness is the weak link** —
the judge marks many in_scope answers partial/incomplete.

### Professor feedback on D1 (drives D2 priorities)
> "For Deliverable 2, focus on strengthening the experimental analysis. In particular, compare
> retrieval strategies, chunking configurations, and evaluation metrics using **quantitative
> evidence rather than qualitative observations alone**. The discussion of MMR and retrieval
> failures is a good starting point, but stronger conclusions should be supported by **systematic
> testing**."

**Translation:** every claim in D2 must be backed by a number from a repeatable experiment, not an
eyeballed observation. The evaluation harness is the top priority.

---

## What this is

**FitRAG** is a domain-specific RAG assistant answering fitness / exercise-science questions for
beginners, grounded in curated PDFs (`data/raw/`). University group project for *CSAI-413 NLP
Applications* (British University in Dubai). Brief: `src/NLPA_Group_Project_2026.pdf`.

Grading: **D1 = 30%** (design + baseline), **D2 = 50%** (experiments + evaluation), **viva/demo = 20%**.
Each member is examined individually, so all work must be **justifiable, explainable, reproducible**.

---

## Deliverable 1 — checklist ✅ (done, 78/100)

- [x] Problem definition (domain, target users, why RAG vs direct LLM)
- [x] Dataset description (10 source PDFs, preprocessing, chunking justification)
- [x] System architecture diagram + component descriptions
- [x] Design justification (embedding choice, retrieval choice, alternatives)
- [x] Baseline RAG implementation (retrieval + generation working end-to-end)
- [x] Initial evaluation (example queries, outputs, early failure cases — *qualitative*)
- [x] Report submitted (`src/Report 1 - NLPA project.docx`)

**Known weakness flagged by professor:** evaluation was qualitative only → fix in D2.

---

## Deliverable 2 — checklist 🔜 (in progress)

### 0. Foundation
- [x] Gold evaluation set built — `data/eval/qa_set.json` (30 Qs: 20 in_scope / 5 ambiguous / 5 out_of_scope, each with `expected_sources` + `reference_answer`)
- [x] **Evaluation harness** — `evaluate(config) → metrics` over all 30 Qs (`src/fitrag_eval.py`, driven by `notebooks/04_evaluation.ipynb`; results → `data/eval/results/<name>.json`)

### 1. Evaluation framework (metrics) — all implemented + validated on baseline
- [x] Retrieval: **Recall@k** (uses `expected_sources`, objective)
- [x] Retrieval: **MRR / Hit-rate**
- [x] Retrieval: **Precision@k**
- [x] Answer: **Correctness** (LLM-as-judge vs `reference_answer`)
- [x] Robustness: **Refusal accuracy** (out_of_scope → refuse; in_scope → don't refuse)
- [x] **Hallucination / faithfulness** score (LLM-judge: claims grounded in context?)
- [x] **Judge can run locally** — `qwen2.5:7b-instruct` via Ollama (`LOCAL_JUDGE` preset), validated against gpt-oss-120b (agreement study: correctness within-1 ≈88%, Pearson r≈0.64; faithfulness within-1 ≈77%, high-mean ceiling effect). Different family from llama generator → bias-avoidance preserved. Unlimited.

### 2. Controlled experiments (vary ONE axis at a time vs baseline)
- [x] Retrieval strategy: similarity vs MMR (sweep `k`, `lambda_mult`) — DONE. Retrieval: `h1_retrieval_sweep.json`. Answer-quality: `local_baseline.json` (MMR) vs `sim_k5_local.json` (similarity), local generator held constant.
- [x] Chunk size: 256 / 512 / 1024 (overlap fixed 50) — indexes rebuilt via `src/build_index.py` → `embeddings/vector_store_cs{N}`. Retrieval sweep `h2_chunksize_sweep.json`; answer-quality `h2_cs{256,512,1024}.json` (cs1024 judging deferred — judge daily cap).
- [x] Embedding model: `multi-qa-MiniLM-L6` vs `all-MiniLM-L6` vs `bge-small`
- [x] Prompt design: strict-grounded vs lenient
- [x] RAG vs direct LLM (no retrieval) — `direct_llm.json`. Direct LLM correctness 0.713 vs RAG 0.637 (+7.6 pp) but out-of-scope refused = 0.000 (answers everything). RAG: out-of-scope refused = 1.000, faithfulness = 0.833 (direct has no faithfulness — no context). Direct LLM's correctness edge is from pretraining knowledge, not grounding. RAG's value: grounding + domain safety + updatability.

### 3. Hypothesis-driven write-ups (hypothesis → setup → results → interpretation)
- [x] H1: "MMR > similarity on Recall@5" — **REJECTED** by systematic testing. At k=5, similarity ≥ MMR on every retrieval metric (in_scope Recall@5 tied at 0.925; similarity better on overall Recall 0.882 vs 0.868, Precision 0.642 vs 0.575, MRR 0.778 vs 0.771). λ-sweep confirms MMR's diversity penalty hurts accuracy (λ=0.8 → Prec 0.650, λ=0.2 → 0.533). MMR's benefit is source diversity, not recall. **Answer-quality (local 8B generator held constant): tied** — Correctness(in_scope) 0.625 both; Faithfulness 0.909 (MMR) vs 0.893 (sim). The retrieval-precision edge of similarity does not improve answers. MMR offers no advantage on retrieval OR answer quality.
- [x] H2: "smaller chunks improve retrieval precision" — **REJECTED**. 256 = *worst* on recall (0.825), MRR (0.689), hit (0.833). 512 = best recall (0.925, validates baseline); 1024 = best precision (0.608) & MRR (0.792). Answer-quality (local qwen judge, all 3 sizes): **faithfulness rises monotonically** 0.716→0.833→0.838 (larger chunk = more complete context = less fabrication); correctness differences small & judge-dependent (within noise N≈20) → conclusion anchored on objective retrieval + faithfulness. Larger chunks carry more complete context; 256 fragments it.
- [x] H3: "better embeddings improve recall" — **REJECTED**. Baseline `multi-qa-MiniLM` already best on in_scope Recall@5 (0.925) and Hit-rate (0.917). `all-MiniLM` (0.875 recall, 0.880 refusal-acc) and `bge-small` (0.875 recall) both worse on retrieval. Reason: `multi-qa-MiniLM` was fine-tuned on 215M QA pairs — QA domain-match outweighs model size. Counter-intuitive: `bge-small` has worst retrieval but best answer correctness (0.675) and faithfulness (0.893) — its higher-precision (though lower-recall) chunks are sufficient for the generator. Baseline embedding choice is now empirically justified, not assumed.
- [x] H4: prompt hypothesis (strict prompt reduces hallucination) — **CONFIRMED**. Faithfulness 0.833 (strict) vs 0.717 (lenient), −11.6 pp. Critical safety finding: lenient prompt answered **all 5 out-of-scope questions** (out_of_scope refused rate 0.000 vs 1.000) — completely breaks the refusal safety net. Correctness slight edge for lenient (0.688 vs 0.637) is artefactual — model draws on training-data knowledge, not retrieved context. Strict prompt validated as correct design choice.

### 4. Failure analysis (example + cause + attempted fix, per type)
- [x] Retrieval failure — Q06: vocabulary collision across 5 NSCA PDFs; fix = larger k / hybrid BM25+dense
- [x] Hallucination — Q11: perfect retrieval but 70B fills in from training knowledge; fix = strict prompt (H4 confirmed)
- [x] Ambiguous query handling — Q22: 6-word query embeds too broadly; fix = query expansion / user clarification
- [x] Irrelevant context — Q20: correct source at rank 4 drowned by 4 noise chunks → wrong refusal; fix = larger chunks + higher MMR λ

### 5. Comparative analysis + results
- [x] Baseline vs improved system (and/or RAG vs direct LLM) — quantitative table
- [x] Results tables + insights write-up — consolidated master table + cross-experiment summary in `04_evaluation.ipynb`

### 6. Reporting
- [ ] D2 report (PDF) with tables, screenshots/evidence
- [ ] AI Usage Disclosure section (tools used, how, what was independent; names/IDs/signatures)

### Bonus (optional, extra credit)
- [ ] Streamlit web UI (`app.py` is currently **empty**)
- [ ] Hybrid / advanced retrieval
- [ ] Hallucination-detection safety mechanism

---

## Pipeline configuration (current baseline)

- **Chunking:** `chunk_size=512`, `chunk_overlap=50`, separators `["\n\n","\n",".", " "]`; drop <100 chars → **2997 chunks**.
- **Embeddings:** `multi-qa-MiniLM-L6-cos-v1` (HF, 384-dim), CPU, normalized.
- **Vector store:** FAISS (L2 over normalized vectors ≈ cosine).
- **Retriever:** MMR — `k=5`, `fetch_k=20`, `lambda_mult=0.5`.
- **LLM (generator):** selectable backend via `config["llm_backend"]`:
  - `groq` (default) → `llama-3.3-70b-versatile` (API; 100k tokens/day free cap).
  - `ollama` → local `llama3.1:8b` on the RTX 3080 (GPU, **unlimited**). Use the `LOCAL_GEN` preset: `evaluate({"name": ..., **fe.LOCAL_GEN})`. A full 30-Q run ≈ 3.5 min.
  - `temperature=0`, `max_tokens=1024` for both.
  - `rag_config.json` once recorded `claude-haiku-4-5`; the chain actually uses the Groq model. Keep config synced with what the chain instantiates.
- **Generator size sensitivity (bonus finding):** Groq 70B vs local 8B (same MMR retrieval + Groq judge) — Correctness(in_scope) 0.588 (70B) vs 0.625 (8B, ≈tied/judge noise); **Faithfulness 0.978 (70B) vs 0.909 (8B)** → the smaller model adds more ungrounded detail. Retrieval + refusal accuracy identical (generator-independent).
- **Generation prompt:** strict grounded — answer only from context, fixed refusal sentence when insufficient, cite sources at end.
- **Eval harness:** `src/fitrag_eval.py`, `evaluate(config) → {config, aggregate, per_question}`; results to `data/eval/results/<name>.json`. Run via `notebooks/04_evaluation.ipynb`.
- **LLM-as-judge:** selectable via `config["judge_backend"]`:
  - `groq` (default) → `openai/gpt-oss-120b` (strong, but **200k tokens/day** cap → ~2 full judged runs/day; was the matrix bottleneck).
  - `ollama` → local `qwen2.5:7b-instruct` (`LOCAL_JUDGE` preset, `format="json"` for reliable parsing). **Unlimited.** Validated vs gpt-oss-120b (agreement study above). Both judges are a different family from the llama generator (bias-avoidance is about family, not provider).
  - Why a judge at all: the brief's §4 mandates measuring **answer correctness** + **hallucination presence** — free-text judgments with no formula. LLM-as-judge is the scalable, reproducible (temp=0), reference-anchored way to quantify them (vs unscalable manual grading or semantics-blind ROUGE).
  - Gemini was the original plan but its free tier caps at **20 requests/day** — unusable. Harness supports `precomputed_answers={id: answer}` to re-judge saved (temp=0) answers without re-spending generator tokens.

---

## Knowledge base & evaluation data

- `data/raw/` — 10 PDFs (WHO guidelines, NSCA position statements, progressive-overload + endurance research, Starting Strength article). `documents.json` = source metadata/citations.
- `data/eval/qa_set.json` — 30-question gold set (see D2 checklist §0). `expected_sources` = PDF stems correct retrieval should surface; powers objective retrieval metrics. out_of_scope items test refusal; ambiguous items test robustness.

---

## Environment & running

- Windows, Python 3.13, venv at `.venv/` → use `.venv/Scripts/python.exe`. Shell: PowerShell primary (Bash tool also available).
- Secrets in `.env` (gitignored): `GROQ_API_KEY` (generator-as-API + judge), `GOOGLE_API_KEY` (unused — Gemini dropped). Loaded via `load_dotenv("../.env")`.
- **Local LLM:** Ollama v0.30.6 installed (`%LOCALAPPDATA%\Programs\Ollama\ollama.exe`), server on `127.0.0.1:11434`. Models pulled: `llama3.1:8b` (generator) + `qwen2.5:7b-instruct` (judge). Python binding: `langchain-ollama` (in venv). Fully-local unlimited run: `evaluate({"name": ..., **fe.LOCAL_GEN, **fe.LOCAL_JUDGE})`. On the 10GB 3080, run generate-then-judge (two-pass) so only one model is resident at a time — the `precomputed_answers` path does this naturally. Headless notebook exec via `nbclient`/`nbformat`.
- Notebook paths are **relative to `notebooks/`** (`../data/...`). Run 01 → 02 → 03 in order; each consumes the prior's output.
- FAISS load uses `allow_dangerous_deserialization=True` (self-produced, trusted index).

## Conventions

- Notebook cells favor explicit, verbose status prints (✅/⚠️, formatted summaries) — match that style for demo/viva legibility.
- Editing notebooks: prefer `NotebookEdit`. On Windows watch encoding — read notebook JSON with `encoding="utf-8"` and run Python with `PYTHONIOENCODING=utf-8` / `-X utf8` to avoid `cp1252` emoji errors.
- Keep `retriever_config.json` / `rag_config.json` as the single source of truth the RAG notebook reads back, so experiments stay reproducible.
- For D2, every experiment should be one call to the evaluation harness with a different config → keeps results comparable and reproducible.

---

## Progress log

- **2026-06-19** — D1 returned, 78/100. Reviewed professor feedback (quantitative evidence needed). Built 30-Q gold eval set (`data/eval/qa_set.json`). Planned D2: evaluation harness first, then controlled-experiment matrix. CLAUDE.md converted to living progress tracker.
- **2026-06-19** — Built the evaluation harness (`src/fitrag_eval.py`) + runner (`notebooks/04_evaluation.ipynb`). All 6 metrics implemented (Recall@k, Precision@k, MRR, Hit-rate, Correctness, Faithfulness, Refusal-accuracy) and validated on a full baseline run → `data/eval/results/baseline.json`. Switched LLM-judge from Gemini (20 req/day free cap) to Groq `gpt-oss-120b`. Baseline: retrieval strong (Recall@k in_scope 0.925, Hit-rate 0.917) but **correctness is the weak link (0.588)**; faithfulness 0.978, refusal-acc 0.96. Auto-surfaced failure cases: Q06/Q22 retrieval miss, Q11 low faithfulness, Q20 wrong refusal.
- **2026-06-19** — Ran **H1 (MMR vs similarity)**. Added `retrieval_only` mode to the harness (free retrieval sweeps). Swept similarity & MMR over k∈{3,5,10} and MMR λ∈{0.2,0.5,0.8} → `data/eval/results/h1_retrieval_sweep.json`, written up in `04_evaluation.ipynb`. **H1 rejected:** similarity ≥ MMR at k=5 on all retrieval metrics; D1's qualitative MMR claim doesn't survive systematic testing (MMR helps diversity, not recall). Generator (`llama-3.3-70b`) then hit Groq's 100k-tokens/day cap.
- **2026-06-19** — **Went local to remove the API bottleneck.** Installed Ollama + `llama3.1:8b` (RTX 3080, 10GB) + `langchain-ollama`; added a selectable generator backend to the harness (`config["llm_backend"]` = `groq`|`ollama`, plus `LOCAL_GEN` preset). Judge stays on Groq `gpt-oss-120b`. Full 30-Q local run ≈ 3.5 min, unlimited. Completed **H1 answer-quality** locally (`local_baseline.json` MMR vs `sim_k5_local.json` similarity): tied — confirms H1 rejected on both retrieval and answer quality. Bonus: 70B-vs-8B shows model size mainly affects **faithfulness** (0.978 → 0.909).
- **2026-06-19** — **H2 (chunk size)** done. Added `src/build_index.py` (rebuilds FAISS at any chunk_size with identical preprocessing; cs512 reproduces 2997 chunks exactly = builder validated). Built cs256/512/1024 indexes. **H2 rejected:** 256 worst on retrieval *and* answer quality; 512 best recall; 1024 best precision/MRR. Hit a new limit: the **judge `gpt-oss-120b` has a 200k tokens/day cap** (~2 full judged runs/day) — exhausted today, so cs1024 answers were generated locally (free) but judging is deferred (re-judge later via `precomputed_answers`). **Recurring issue:** the Groq judge is now the matrix bottleneck. **Next:** decide local-judge vs pace runs; then H3 (embedding model) / H4 (prompt).
- **2026-06-19** — **Consolidated results table done.** Master table in `04_evaluation.ipynb` covering all 15 configs across H1–H4 + RAG vs direct LLM, with cross-experiment insights. Key finding: 3/4 hypotheses rejected — baseline is well-chosen; generator size (70B vs 8B) is the largest remaining lever on correctness/faithfulness. Only D2 report PDF remains.
- **2026-06-19** — **Failure analysis done.** All 4 types written up in `04_evaluation.ipynb` with example, explanation, attempted fix: (1) Q06 retrieval failure — vocabulary collision, fix = larger k/hybrid; (2) Q11 hallucination — 70B overrides strict prompt, fix = post-generation faithfulness check; (3) Q22 ambiguous query — 6-word query underspecified, fix = query expansion; (4) Q20 irrelevant context — correct source drowned by noise, fix = larger chunks + higher MMR λ.
- **2026-06-19** — **RAG vs Direct LLM done.** `direct_llm.json`. Direct LLM correctness 0.713 (vs RAG 0.637) but answered ALL out-of-scope questions (refused rate 0.000 vs 1.000 for RAG). No faithfulness metric (no context). RAG wins on grounding, domain safety, and updatability — correctness gap is from strict-prompt cost, not architecture failure. Notebook section added.
- **2026-06-19** — **H4 (prompt design) done.** Ran lenient prompt variant (`h4_lenient.json`). **H4 confirmed:** strict prompt faithfulness 0.833 vs lenient 0.717 (−11.6 pp). Critical safety finding: lenient prompt answered all 5 out-of-scope questions (out_of_scope refused = 0.000 vs 1.000 for strict) — completely breaks domain boundary. Correctness edge for lenient (0.688 vs 0.637) is artefactual (model uses training-data knowledge, not KB). H4 cells added to `04_evaluation.ipynb`. All 4 hypotheses done — RAG vs direct LLM + failure analysis next.
- **2026-06-19** — **H3 (embedding model) done.** Ran `all-MiniLM-L6-v2` and `bge-small-en-v1.5` full evaluations (local llama gen + qwen judge). Retrieval sweep already in `h3_embedding_sweep.json`; answer-quality in `h3_allMiniLM.json` + `h3_bge.json`. **H3 rejected:** baseline `multi-qa-MiniLM` has best in_scope Recall (0.925) and Hit-rate (0.917) — QA-specific training outweighs model size. `bge-small` best correctness (0.675) and faithfulness (0.893) despite worst retrieval recall — counter-intuitive but within noise. H3 cells added to `04_evaluation.ipynb`. Next: H4 (strict vs lenient prompt).
- **2026-06-19** — **Judge moved local too → fully unlimited offline pipeline.** Added selectable `judge_backend` (`groq`|`ollama`) + `LOCAL_JUDGE` preset (`qwen2.5:7b-instruct`, `format="json"`). Validated the 7B judge against gpt-oss-120b on already-judged answers: **correctness within-1 ≈88%, Pearson r≈0.64; faithfulness within-1 ≈77%** (high-mean ceiling effect lowers r). 7B judge is slightly more lenient on correctness / stricter on faithfulness — systematic, so use one judge consistently per comparison. Re-judged all 3 H2 sizes with qwen for consistency → completed H2 answer-quality: faithfulness monotonic with chunk size (0.716→0.833→0.838); correctness within-noise. Both gen+judge now local on the RTX 3080. **Next:** H3 (embedding model), H4 (prompt) — all runnable unlimited.
