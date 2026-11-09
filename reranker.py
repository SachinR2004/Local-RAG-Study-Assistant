"""
reranker.py

Reorders retrieved document chunks using a cross-encoder.
"""

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _get_model(self):
        """Load the cross-encoder only when first needed."""
        if self._model is None:
            self._model = CrossEncoder(
                self.model_name,
                device=self.device
            )

        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int | None = None,
        text_key: str = "text"
    ) -> list[dict]:
        """
        Rerank retrieved documents against the query.

        Each document must contain text_key.

        Returns copied document dictionaries with an additional
        'rerank_score', sorted from most to least relevant.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not documents:
            return []

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        for document in documents:
            if text_key not in document:
                raise KeyError(
                    f"Document is missing the '{text_key}' field."
                )

        model = self._get_model()

        pairs = [
            [query, document[text_key]]
            for document in documents
        ]

        scores = model.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False
        )

        reranked = []

        for document, score in zip(documents, scores):
            result = document.copy()
            result["rerank_score"] = float(score)
            reranked.append(result)

        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked