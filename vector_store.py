import os
import pickle

import faiss
import numpy as np


class VectorStore:

    def __init__(self):
        self.index = None
        self.chunks = []
        self.sources = []
        self.metadata = []

    def build(self, embeddings, chunks, sources, metadata=None):
        """Create a new FAISS index."""

        if len(chunks) != len(sources):
            raise ValueError("Chunks and sources must have equal lengths.")

        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks must have equal lengths.")

        if not chunks:
            raise ValueError("Cannot build an empty vector index.")

        if metadata is None:
            metadata = [{} for _ in chunks]

        if len(metadata) != len(chunks):
            raise ValueError("Metadata and chunks must have equal lengths.")

        embeddings = np.asarray(embeddings, dtype="float32")

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        self.chunks = list(chunks)
        self.sources = list(sources)
        self.metadata = list(metadata)

    def add(self, embeddings, chunks, sources, metadata=None):
        """Add new embeddings and metadata to the existing index."""

        if len(chunks) != len(sources):
            raise ValueError("Chunks and sources must have equal lengths.")

        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks must have equal lengths.")

        if not chunks:
            return

        if metadata is None:
            metadata = [{} for _ in chunks]

        if len(metadata) != len(chunks):
            raise ValueError("Metadata and chunks must have equal lengths.")

        embeddings = np.asarray(embeddings, dtype="float32")

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if self.index is None:
            self.build(embeddings, chunks, sources, metadata)
            return

        if embeddings.shape[1] != self.index.d:
            raise ValueError("Embedding dimension does not match the index.")

        self.index.add(embeddings)

        self.chunks.extend(chunks)
        self.sources.extend(sources)
        self.metadata.extend(metadata)

    def search(self, query_embedding, top_k=3):
        """Search the FAISS index for similar vectors."""

        if self.index is None or self.index.ntotal == 0:
            return None, None

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        if query_embedding.ndim != 2:
            raise ValueError("Query embedding must be a 2D array.")

        if query_embedding.shape[1] != self.index.d:
            raise ValueError("Query embedding dimension does not match index.")

        return self.index.search(query_embedding, top_k)

    def save(self, path):
        """Save FAISS index, chunks, sources and metadata."""

        if self.index is None:
            raise ValueError("Cannot save an empty index.")

        os.makedirs(path, exist_ok=True)

        faiss.write_index(
            self.index,
            os.path.join(path, "index.faiss")
        )

        with open(os.path.join(path, "meta.pkl"), "wb") as file:
            pickle.dump(
                {
                    "chunks": self.chunks,
                    "sources": self.sources,
                    "metadata": self.metadata
                },
                file
            )

    def load(self, path):
        """Load FAISS index and associated metadata."""

        index_path = os.path.join(path, "index.faiss")
        metadata_path = os.path.join(path, "meta.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(index_path)

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(metadata_path)

        index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as file:
            stored_data = pickle.load(file)

        chunks = stored_data["chunks"]
        sources = stored_data["sources"]

        # Support older indexes created before metadata was introduced.
        metadata = stored_data.get(
            "metadata",
            [{} for _ in chunks]
        )

        if index.ntotal != len(chunks):
            raise ValueError("FAISS index and chunks are inconsistent.")

        if len(chunks) != len(sources):
            raise ValueError("Chunks and sources are inconsistent.")

        if len(chunks) != len(metadata):
            raise ValueError("Chunks and metadata are inconsistent.")

        self.index = index
        self.chunks = chunks
        self.sources = sources
        self.metadata = metadata
