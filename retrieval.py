
"""
retrieval.py

Keyword retrieval using BM25 and ranking fusion
using Reciprocal Rank Fusion (RRF).
"""

import re
from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self):
        self._bm25 = None
        self._chunks = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Convert text into searchable lowercase tokens."""
        return re.findall(
            r"[a-z0-9]+(?:[./_-][a-z0-9]+)*",
            text.lower()
        )

    def build(self, chunks: list[str]):
        """Build the BM25 index."""
        if not chunks:
            raise ValueError("Cannot build BM25 index from empty chunks.")

        if any(not isinstance(chunk, str) for chunk in chunks):
            raise TypeError("Every chunk must be a string.")

        tokenized_chunks = [
            self.tokenize(chunk)
            for chunk in chunks
        ]

        self._chunks = list(chunks)
        self._bm25 = BM25Okapi(tokenized_chunks)

    def search(self, query: str, top_k: int = 5):
        """Return (chunk_index, BM25_score) pairs."""
        if self._bm25 is None:
            raise RuntimeError("Build the BM25 index before searching.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for index in ranked_indices:
            score = float(scores[index])

            if score <= 0:
                continue

            results.append((index, score))

            if len(results) >= top_k:
                break

        return results


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int = 60
) -> list[tuple[int, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    Each ranking contains document indices ordered from
    most relevant to least relevant.

    Returns:
        [(document_index, fused_score), ...]
    """
    if k <= 0:
        raise ValueError("RRF k must be greater than zero.")

    fused_scores = {}

    for ranking in rankings:
        for rank, document_index in enumerate(ranking, start=1):

            fused_scores[document_index] = (
                fused_scores.get(document_index, 0.0)
                + 1.0 / (k + rank)
            )

    return sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )
