"""
rag_utils.py
"""

import os

# Prevent OpenMP runtime conflict on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import torch BEFORE faiss / sentence_transformers to set up runtime memory correctly
import torch
import pickle
from dataclasses import dataclass, field

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MIN_SIMILARITY = 0.35  # below this, we treat the match as "not found"


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a typed/printed PDF (not scanned/handwritten)."""
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 220, overlap: int = 40) -> list[str]:
    """Split text into overlapping word-count chunks."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        start += step
    return chunks


@dataclass
class RAGIndex:
    """Wraps embedding + FAISS index + the chunk text each vector maps to."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    _model: SentenceTransformer = field(default=None, repr=False)
    _index: faiss.Index = field(default=None, repr=False)
    chunks: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)  # which doc each chunk came from

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build(self, chunks: list[str], sources: list[str]):
        """Embed all chunks and build a FAISS index from scratch."""
        assert len(chunks) == len(sources)
        model = self._get_model()
        embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine sim
        index.add(embeddings)

        self._index = index
        self.chunks = list(chunks)
        self.sources = list(sources)

    def add(self, chunks: list[str], sources: list[str]):
        """Add more chunks to an existing index (e.g. a second uploaded PDF)."""
        if self._index is None:
            self.build(chunks, sources)
            return
        model = self._get_model()
        embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype="float32")
        self._index.add(embeddings)
        self.chunks.extend(chunks)
        self.sources.extend(sources)

    def query(self, question: str, top_k: int = 3):
        """Return top_k (chunk, source, score) tuples for a question.

        Returns an empty list if nothing clears MIN_SIMILARITY.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        model = self._get_model()
        q_emb = model.encode([question], normalize_embeddings=True, show_progress_bar=False)
        q_emb = np.asarray(q_emb, dtype="float32")

        scores, idxs = self._index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            if score < MIN_SIMILARITY:
                continue
            results.append((self.chunks[idx], self.sources[idx], float(score)))
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self._index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "meta.pkl"), "wb") as f:
            pickle.dump({"chunks": self.chunks, "sources": self.sources}, f)

    def load(self, path: str):
        self._index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        self.chunks = meta["chunks"]
        self.sources = meta["sources"]