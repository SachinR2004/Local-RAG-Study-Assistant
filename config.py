"""
Central configuration for the Local RAG Study Assistant.
"""

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Text chunking
CHUNK_SIZE = 220
CHUNK_OVERLAP = 40

# Retrieval settings
MIN_SIMILARITY = 0.20
TOP_K = 3

# Text generation
GENERATION_MODEL = "google/flan-t5-base"

# Device configuration
DEVICE = "cpu"