import os

# Prevent OpenMP runtime conflicts on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize PyTorch before Sentence Transformers
import torch
import numpy as np

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


class EmbeddingModel:

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        """Load the embedding model only when required."""

        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:

        if not texts:
            return np.empty((0, 0), dtype="float32")

        model = self._get_model()

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return np.asarray(embeddings, dtype="float32")
