# 🏋️ FitRAG — Fitness AI Assistant
> A Retrieval-Augmented Generation (RAG) system for answering fitness and exercise science questions, grounded in curated knowledge sources.

---

## 📌 Project Overview

FitRAG is a domain-specific AI assistant designed to answer fitness-related questions for beginners and general users. The system retrieves relevant information from a curated knowledge base of fitness guides and sports science research papers, then uses a large language model to generate grounded, accurate responses.


---

## 🗂️ Project Structure

```
fitrag/
│
├── data/
│   ├── raw/                  # Original PDF documents
│   └── processed/            # Cleaned and chunked text
│
├── embeddings/
│   └── vector_store/         # FAISS / ChromaDB index files
│
├── notebooks/
│   ├── baseline_demo.ipynb   # Baseline RAG system demo
│   └── experiments.ipynb     # Deliverable 2 experiments
│
├── src/
│   ├── preprocessing.py      # Document loading and chunking
│   ├── embeddings.py         # Embedding model and indexing
│   ├── retrieval.py          # Vector store retrieval logic
│   ├── generation.py         # LLM prompt and response generation
│   └── evaluation.py         # Evaluation pipeline
│
├── app.py                    # Streamlit web interface
├── README.md
└── requirements.txt          # Python dependencies      
```

---

## ⚙️ System Architecture

```
Documents (PDFs)
      │
      ▼
Preprocessing & Chunking
      │
      ▼
Text Embeddings (sentence-transformers)
      │
      ▼
Vector Store Index (FAISS / ChromaDB)
      │
      ▼
Retrieval (top-k similarity search)
      │
      ▼
LLM Response Generation (with retrieved context)
      │
      ▼
Evaluated Output
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip package manager

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fitrag.git
cd fitrag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your documents

Place your fitness PDF documents inside the `data/raw/` folder.

### 4. Build the vector index

```bash
python src/preprocessing.py
python src/embeddings.py
```

### 5. Run the RAG system

```bash
python src/generation.py
```

### 6. (Optional) Launch the web interface

```bash
streamlit run app.py
```

---

## 📦 Dependencies

```
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 📚 Knowledge Base Sources

| # | Title | Type | Source |
|---|-------|------|--------|
| 1 | WHO GUIDELINES ON PHYSICAL ACTIVITY AND SEDENTARY BEHAVIOUR | Health Guideline | World Health Organization |
| 2 | Increasing Anaerobic Endurance Using Strength Endurance Training and Continuous Running | Sports Science Paper | JUMORA / DOI |
| 3 | Progressive Overload in Long-Term Exercise Interventions Targeting Executive Function | Scholarly Review | Kinesiology Review |
| 4 | Youth Resistance Training: Updated Position Statement Paper from the NSCA | Position Statement | nsca.com |
| 5 | NSCA Position Statement on Long-Term Athletic Development | Position Statement | nsca.com |
| 6 | Resistance Training for Older Adults: Position Statement from the NSCA | Position Statement | nsca.com |
| 7 | NSCA Position Statement on Weightlifting for Sports Performance | Position Statement | nsca.com |
| 8 | NSCA Strength and Conditioning Professional Standards and Guidelines | Professional Standards | nsca.com |
| 9 | Who Wants to Be a Novice? You Do | Training Article | startingstrength.com |
| 10 | Progressive Overload (CEU Quiz) | Training Guide | ncsf.org |

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Retrieval Relevance | Are the retrieved chunks relevant to the query? |
| Answer Correctness | Is the generated answer factually accurate? |
| Hallucination Rate | Does the answer contain unsupported claims? |
| Robustness | How does the system handle ambiguous or out-of-scope queries? |