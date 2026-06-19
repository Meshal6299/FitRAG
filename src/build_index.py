"""
Reusable FAISS index builder for chunk-size experiments (D2 / H2).

Replicates the notebook 01 (preprocess + chunk) and 02 (embed + index) pipeline
in one function so we can rebuild the vector store at any chunk size with
identical preprocessing — the only axis that changes is ``chunk_size``.

    python -m src.build_index            # builds 256 / 512 / 1024 (overlap=50)
    python -m src.build_index 256 1024   # builds just those sizes

Each index is saved to ``embeddings/vector_store_cs<size>/`` and an
``evaluate({..., "index_path": "embeddings/vector_store_cs256"})`` run can then
point at it. Overlap is held fixed (default 50) so chunk_size is the only varied
axis. The embedding model and the <100-char drop match the baseline exactly.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

EMBED_MODEL = "multi-qa-MiniLM-L6-cos-v1"   # same as baseline
MIN_CHUNK_LENGTH = 100                        # drop tiny chunks (matches notebook 01)
SEPARATORS = ["\n\n", "\n", ".", " "]        # matches notebook 01


def clean_text(text: str) -> str:
    """Same cleaning as notebook 01."""
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\{3,\}", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def load_documents() -> List:
    from langchain_community.document_loaders import PyPDFLoader

    docs = []
    pdfs = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".pdf"))
    for fname in pdfs:
        loaded = PyPDFLoader(str(RAW_DIR / fname)).load()
        docs.extend(loaded)
        print(f"  loaded {fname:<55} pages={len(loaded)}")
    for d in docs:
        d.page_content = clean_text(d.page_content)
    print(f"  total pages: {len(docs)}")
    return docs


def build_index(chunk_size: int, chunk_overlap: int = 50, documents=None,
                embed_model: str = EMBED_MODEL, index_rel: str = None) -> str:
    """Chunk -> embed -> FAISS -> save. Returns the index path (rel to repo root).

    ``embed_model`` selects the HF embedding model (for H3 embedding-model
    experiments). ``index_rel`` overrides the output path; defaults to
    ``embeddings/vector_store_cs{chunk_size}`` (chunk-size experiments)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    if documents is None:
        documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    before = len(chunks)
    chunks = [c for c in chunks if len(c.page_content) >= MIN_CHUNK_LENGTH]

    if index_rel is None:
        index_rel = f"embeddings/vector_store_cs{chunk_size}"
    index_path = REPO_ROOT / index_rel

    print(f"\n[chunk_size={chunk_size}, overlap={chunk_overlap}, embed={embed_model}] "
          f"chunks: {before} -> {len(chunks)} after <{MIN_CHUNK_LENGTH}-char drop")

    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    t = time.time()
    vs = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vs.save_local(str(index_path))
    print(f"  built + saved {vs.index.ntotal} vectors in {time.time() - t:.0f}s -> {index_rel}")
    return index_rel


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or [256, 512, 1024]
    docs = load_documents()  # load PDFs once, reuse across sizes
    for size in sizes:
        build_index(size, documents=docs)
    print("\nDone.")
