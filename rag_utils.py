
import os
import re

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dataclasses import dataclass, field

from config import MIN_SIMILARITY, EMBEDDING_MODEL

from ingestion import extract_pdf_pages, extract_text_from_pdf

from chunking import (
    chunk_text,
    chunk_pages,
    chunk_pages_by_section
)

from embeddings import EmbeddingModel

from vector_store import VectorStore

from retrieval import BM25Retriever, reciprocal_rank_fusion
from reranker import CrossEncoderReranker


# Lower threshold specifically for explicit multi-activity comparisons.
COMPARISON_MIN_SIMILARITY = 0.10


def extract_activity_filters(question: str) -> list[str]:
    """
    Detect single and grouped activity references.

    Examples:
        Explain Activity 3
        Compare Activity 2 and Activity 4
        Explain activities 1 and 3
        Explain Activity 2, 3 and 4
    """

    question = question.lower()

    activity_numbers = []

    pattern = r"\bactivit(?:y|ies)\b"

    for match in re.finditer(pattern, question):

        remaining = question[match.end():]

        number_match = re.match(
            r"\s*[-:#]?\s*(\d+)",
            remaining
        )

        if not number_match:
            continue

        activity_numbers.append(number_match.group(1))

        remaining = remaining[number_match.end():]

        while True:

            next_match = re.match(
                r"\s*(?:,|and|&)\s*(\d+)",
                remaining
            )

            if not next_match:
                break

            activity_numbers.append(next_match.group(1))

            remaining = remaining[next_match.end():]

    return list(dict.fromkeys(
        f"Activity {number}" for number in activity_numbers
    ))


def _is_heading_only_chunk(text: str) -> bool:
    """
    Identify chunks that contain only a title/slide heading rather than
    explanatory study content.
    """

    text = re.sub(r"\s+", " ", str(text)).strip()

    if not text:
        return True

    # Remove the activity prefix for a cleaner title-length check.
    cleaned = re.sub(
        r"^Activity\s+\d+\s*[—:-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # A short fragment without sentence-ending punctuation is likely a heading.
    word_count = len(cleaned.split())

    has_sentence = bool(
        re.search(r"[.!?](?:[”’\"')\]]*)?$", cleaned)
    )

    if word_count <= 16 and not has_sentence:
        return True

    # Catch longer slide-title chunks, but retain chunks containing prose.
    has_activity_prefix = bool(re.match(
        r"^Activity\s+\d+\s*[—:-]",
        text,
        re.IGNORECASE,
    ))

    has_slide_label = bool(re.search(
        r"\bSlide\s+\d+\s*:",
        text,
        re.IGNORECASE,
    ))

    has_sentence_in_body = bool(re.search(r"[.!?]\s+[A-Z]", text))

    return has_activity_prefix and has_slide_label and not has_sentence_in_body


@dataclass
class RAGIndex:

    model_name: str = EMBEDDING_MODEL

    _embedder: EmbeddingModel = field(
        default=None,
        repr=False
    )

    _vector_store: VectorStore = field(
        default_factory=VectorStore,
        repr=False
    )

    _bm25_retriever: BM25Retriever = field(
        default_factory=BM25Retriever,
        repr=False
    )

    _reranker: CrossEncoderReranker = field(
        default_factory=CrossEncoderReranker,
        repr=False
    )

    @property
    def _index(self):
        return self._vector_store.index

    @property
    def chunks(self):
        return self._vector_store.chunks

    @property
    def sources(self):
        return self._vector_store.sources

    @property
    def metadata(self):
        return self._vector_store.metadata

    def _get_embedder(self):
        """Load the embedding model lazily."""

        if self._embedder is None:
            self._embedder = EmbeddingModel(self.model_name)

        return self._embedder

    def build(self, chunks, sources, metadata=None):
        """Create a new FAISS and BM25 index."""

        if len(chunks) != len(sources):
            raise ValueError(
                "Chunks and sources must have equal lengths."
            )

        if not chunks:
            raise ValueError(
                "Cannot build an index with no chunks."
            )

        if metadata is None:
            metadata = [{} for _ in chunks]

        if len(metadata) != len(chunks):
            raise ValueError(
                "Metadata and chunks must have equal lengths."
            )

        embeddings = self._get_embedder().encode(chunks)

        self._vector_store.build(
            embeddings,
            chunks,
            sources,
            metadata
        )

        # Build keyword index using the complete chunk collection.
        self._bm25_retriever.build(self.chunks)

    def add(self, chunks, sources, metadata=None):
        """Add chunks to both retrieval indexes."""

        if not chunks:
            return

        if len(chunks) != len(sources):
            raise ValueError(
                "Chunks and sources must have equal lengths."
            )

        if metadata is None:
            metadata = [{} for _ in chunks]

        if len(metadata) != len(chunks):
            raise ValueError(
                "Metadata and chunks must have equal lengths."
            )

        embeddings = self._get_embedder().encode(chunks)

        self._vector_store.add(
            embeddings,
            chunks,
            sources,
            metadata
        )

        # Rebuild BM25 so it includes all indexed documents.
        self._bm25_retriever.build(self.chunks)

    def add_pages(self, pages, source):
        """
        Chunk extracted PDF pages and preserve metadata.
        """

        if not pages:
            return 0

        document_chunks = chunk_pages_by_section(pages)

        if not document_chunks:
            return 0

        chunks = [
            item["text"]
            for item in document_chunks
        ]

        metadata = [
            {
                "page": item["page"],
                "chunk_index": item["chunk_index"],
                "section": item["section"],
                "activity": item["activity"]
            }
            for item in document_chunks
        ]

        sources = [source] * len(chunks)

        self.add(chunks, sources, metadata)

        return len(chunks)

    def query_with_metadata(self, question, top_k=3):
        """
        Hybrid retrieval with activity-aware filtering, content-aware
        candidate selection, duplicate removal, and balanced reranking.
        """

        if self._index is None or self._index.ntotal == 0:
            return []

        if top_k <= 0:
            return []

        total_chunks = self._index.ntotal

        activity_filters = extract_activity_filters(question)
        is_comparison = len(activity_filters) > 1

        search_k = (
            total_chunks
            if activity_filters
            else min(total_chunks, max(top_k * 5, 15))
        )

        # 1. FAISS semantic retrieval.
        query_embedding = self._get_embedder().encode([question])

        faiss_scores, faiss_indices = self._vector_store.search(
            query_embedding,
            search_k
        )

        faiss_ranking = [
            int(idx) for idx in faiss_indices[0] if idx != -1
        ]

        semantic_scores = {
            int(idx): float(score)
            for score, idx in zip(faiss_scores[0], faiss_indices[0])
            if idx != -1
        }

        # 2. BM25 keyword retrieval.
        bm25_results = self._bm25_retriever.search(
            question,
            top_k=search_k
        )

        bm25_ranking = [
            index for index, score in bm25_results
        ]

        bm25_scores = {
            index: score for index, score in bm25_results
        }

        # 3. Reciprocal Rank Fusion.
        fused_results = reciprocal_rank_fusion(
            [faiss_ranking, bm25_ranking]
        )

        hybrid_ranks = {
            index: rank
            for rank, (index, score) in enumerate(fused_results, start=1)
        }

        # 4. Build candidates and remove duplicate chunks.
        candidates = []
        comparison_candidates = []

        seen_chunks = set()

        threshold = (
            COMPARISON_MIN_SIMILARITY
            if is_comparison
            else MIN_SIMILARITY
        )

        for idx, fused_score in fused_results:

            chunk_metadata = self.metadata[idx]
            activity = chunk_metadata.get("activity")

            if activity_filters and activity not in activity_filters:
                continue

            chunk_text_value = self.chunks[idx]
            source = self.sources[idx]
            page = chunk_metadata.get("page")
            chunk_index = chunk_metadata.get("chunk_index")

            # Stable identity for duplicate detection.
            if chunk_index is not None:
                chunk_identity = (
                    source,
                    page,
                    chunk_index
                )
            else:
                normalized_text = re.sub(
                    r"\s+",
                    " ",
                    chunk_text_value
                ).strip()

                chunk_identity = (
                    source,
                    page,
                    normalized_text
                )

            if chunk_identity in seen_chunks:
                continue

            seen_chunks.add(chunk_identity)

            semantic_score = semantic_scores.get(idx, 0.0)
            keyword_score = bm25_scores.get(idx, 0.0)

            item = {
                "text": chunk_text_value,
                "source": source,
                "page": page,
                "chunk_index": chunk_index,
                "section": chunk_metadata.get("section"),
                "activity": activity,
                "score": semantic_score,
                "bm25_score": keyword_score,
                "rrf_score": fused_score,
                "_document_index": idx,
                "_hybrid_rank": hybrid_ranks[idx]
            }

            if is_comparison:
                comparison_candidates.append(item)

            if semantic_score >= threshold or keyword_score > 0:
                candidates.append(item)

        # 5. Select the candidate pool.
        if is_comparison:
            reranking_pool = comparison_candidates
        else:
            reranking_pool = candidates[:max(top_k * 3, 10)]

        if not reranking_pool:
            return []

        # 6. Detect actual heading fragments.
        def is_heading_only(item):
            return _is_heading_only_chunk(item["text"])

        substantive_activities = {
            item.get("activity")
            for item in reranking_pool
            if not is_heading_only(item)
        }

        filtered_pool = [
            item for item in reranking_pool
            if (
                not is_heading_only(item)
                or item.get("activity") not in substantive_activities
            )
        ]

        if not filtered_pool:
            filtered_pool = reranking_pool

        # 7. Cross-encoder reranking.
        reranked_candidates = self._reranker.rerank(
            question,
            filtered_pool,
            top_k=None
        )

        if not reranked_candidates:
            return []

        # 8. Combine hybrid and reranker ranks.
        # Hybrid evidence receives more weight.
        rerank_ranks = {
            item["_document_index"]: rank
            for rank, item in enumerate(reranked_candidates, start=1)
        }

        for item in reranked_candidates:

            hybrid_rank = item["_hybrid_rank"]
            rerank_rank = rerank_ranks[item["_document_index"]]

            item["hybrid_rank"] = hybrid_rank

            item["combined_score"] = (
                0.75 / (60 + hybrid_rank)
                + 0.25 / (60 + rerank_rank)
            )

        reranked_candidates.sort(
            key=lambda item: item["combined_score"],
            reverse=True
        )

        # 9. Select final results.
        # Comparisons reserve results for each requested activity.
        if not is_comparison:

            results = reranked_candidates[:top_k]

        else:

            selected = []
            selected_indices = set()

            per_activity_quota = max(
                1,
                top_k // len(activity_filters)
            )

            for requested_activity in activity_filters:

                activity_candidates = [
                    item for item in reranked_candidates
                    if item["activity"] == requested_activity
                ]

                for item in activity_candidates[:per_activity_quota]:

                    idx = item["_document_index"]

                    if idx not in selected_indices:
                        selected.append(item)
                        selected_indices.add(idx)

            # Fill remaining slots using the combined ranking.
            for item in reranked_candidates:

                if len(selected) >= top_k:
                    break

                idx = item["_document_index"]

                if idx not in selected_indices:
                    selected.append(item)
                    selected_indices.add(idx)

            selected.sort(
                key=lambda item: item["combined_score"],
                reverse=True
            )

            results = selected[:top_k]

        # 10. Remove internal fields while preserving diagnostics.
        for item in results:
            item.pop("_document_index", None)
            item.pop("_hybrid_rank", None)

        return results

    def query(self, question, top_k=3):
        """
        Backward-compatible tuple-based retrieval.
        """

        results = self.query_with_metadata(
            question,
            top_k
        )

        return [
            (
                item["text"],
                item["source"],
                item["score"]
            )
            for item in results
        ]

    def save(self, path):
        """Persist the FAISS index and metadata."""

        self._vector_store.save(path)

    def load(self, path):
        """
        Load the FAISS index and metadata.
        Rebuild BM25 from the loaded chunks.
        """

        self._vector_store.load(path)

        if self.chunks:
            self._bm25_retriever.build(self.chunks)
