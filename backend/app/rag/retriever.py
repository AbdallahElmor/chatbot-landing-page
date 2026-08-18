import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from app.rag.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1)
    b = np.array(v2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def tokenize_text(text: str) -> List[str]:
    """Tokenize English and Arabic text into lowercase words."""
    if not text:
        return []
    return re.findall(r'[\w]+', text.lower(), re.UNICODE)


def get_chunk_text(chunk: Dict[str, Any]) -> str:
    """Builds a single indexable text blob from a chunk, regardless of schema.

    Supports the bilingual Synkro schema (title, text_en, text_ar,
    question_en, question_ar, answer_en, answer_ar, retrieval_keywords)
    as well as a generic 'content' field for other document schemas.
    """
    parts = []
    for field in (
        "title",
        "text_en",
        "text_ar",
        "question_en",
        "question_ar",
        "answer_en",
        "answer_ar",
    ):
        value = chunk.get(field)
        if value:
            parts.append(str(value))

    keywords = chunk.get("retrieval_keywords")
    if keywords:
        parts.append(" ".join(keywords))

    # Fallback for generic/other schemas that use a plain "content" field
    if not parts and chunk.get("content"):
        parts.append(str(chunk["content"]))

    return " ".join(parts)


def get_chunk_source_label(chunk: Dict[str, Any]) -> str:
    """Best-effort human-readable source label for a chunk, across schemas."""
    return (
        chunk.get("source")
        or chunk.get("title")
        or chunk.get("section")
        or chunk.get("topic")
        or chunk.get("id")
        or "unknown"
    )


class HybridRetriever:
    """In-memory Hybrid Search Retriever combining BM25 keyword search and
    BAAI/bge-m3 dense embeddings with Reciprocal Rank Fusion (RRF)."""

    def __init__(self, embedding_service: EmbeddingService = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.chunks: List[Dict[str, Any]] = []
        self.bm25_model = None
        self.corpus_tokens: List[List[str]] = []

    def set_chunks(self, chunks: List[Dict[str, Any]]):
        """Sets chunks list and builds BM25 index and precomputed embeddings."""
        self.chunks = chunks
        self._build_indices()

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Adds new chunks and rebuilds indices."""
        for chunk in chunks:
            if "embedding" not in chunk or not chunk["embedding"]:
                chunk["embedding"] = self.embedding_service.get_embedding(
                    get_chunk_text(chunk)
                )
            self.chunks.append(chunk)
        self._build_indices()

    def load_chunks_file(self, file_path: Path):
        """Loads precomputed/pre-chunked entries from a JSON file."""
        path = Path(file_path)
        if not path.exists():
            logger.warning(
                f"Knowledge file not found at {path}. Retriever initialized with 0 chunks."
            )
            self.chunks = []
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
                if isinstance(loaded_data, list):
                    self.chunks = loaded_data
                else:
                    self.chunks = [loaded_data]
            logger.info(f"Loaded {len(self.chunks)} chunks from {path}")
            self._build_indices()
        except Exception as e:
            logger.error(f"Error loading chunks file {path}: {e}")
            self.chunks = []

    def _build_indices(self):
        """Builds BM25 index and precomputes missing embeddings."""
        if not self.chunks:
            self.bm25_model = None
            self.corpus_tokens = []
            return

        # Build BM25 corpus tokens from the combined bilingual text of each chunk
        self.corpus_tokens = [tokenize_text(get_chunk_text(c)) for c in self.chunks]

        try:
            from rank_bm25 import BM25Okapi
            self.bm25_model = BM25Okapi(self.corpus_tokens)
            logger.info("BM25 index successfully built using rank_bm25 Okapi.")
        except ImportError:
            logger.warning(
                "rank_bm25 not installed. Using simple fallback term overlap retriever."
            )
            self.bm25_model = None

        # Precompute embeddings for chunks missing embeddings
        texts_to_embed = []
        indices_to_embed = []
        for i, chunk in enumerate(self.chunks):
            if "embedding" not in chunk or not chunk["embedding"]:
                texts_to_embed.append(get_chunk_text(chunk))
                indices_to_embed.append(i)

        if texts_to_embed:
            logger.info(f"Computing embeddings for {len(texts_to_embed)} chunks...")
            embeddings = self.embedding_service.get_embeddings(texts_to_embed)
            for idx, emb in zip(indices_to_embed, embeddings):
                self.chunks[idx]["embedding"] = emb

    def _compute_bm25_scores(self, query_tokens: List[str]) -> List[float]:
        """Computes BM25 lexical scores for all chunks."""
        if not self.chunks:
            return []

        if self.bm25_model is not None and query_tokens:
            return self.bm25_model.get_scores(query_tokens).tolist()

        # Fallback simple token overlap scoring if rank_bm25 not present
        # or query produced no tokens (still return one score per chunk).
        query_set = set(query_tokens)
        scores = []
        for doc_tokens in self.corpus_tokens:
            if not doc_tokens or not query_set:
                scores.append(0.0)
            else:
                overlap = sum(1 for token in doc_tokens if token in query_set)
                scores.append(float(overlap) / (len(doc_tokens) + 1.0))
        return scores

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """Retrieves top_k relevant chunks matching query string using
        Hybrid (BM25 + dense embeddings) Reciprocal Rank Fusion search."""
        if not self.chunks:
            return []

        num_docs = len(self.chunks)

        query_tokens = tokenize_text(query)
        bm25_scores = self._compute_bm25_scores(query_tokens)

        # Defensive guard: bm25_scores must line up 1:1 with self.chunks,
        # otherwise rank fusion below will silently misalign.
        if len(bm25_scores) != num_docs:
            logger.error(
                f"BM25 score count ({len(bm25_scores)}) does not match chunk "
                f"count ({num_docs}); rebuilding indices."
            )
            self._build_indices()
            bm25_scores = self._compute_bm25_scores(query_tokens)
            if len(bm25_scores) != num_docs:
                # Still mismatched — bail out safely rather than crash.
                bm25_scores = [0.0] * num_docs

        # Compute dense similarity scores
        query_embedding = self.embedding_service.get_embedding(query)
        dense_scores = []
        for chunk in self.chunks:
            embedding = chunk.get("embedding")
            if not embedding:
                embedding = self.embedding_service.get_embedding(get_chunk_text(chunk))
                chunk["embedding"] = embedding
            score = cosine_similarity(query_embedding, embedding)
            dense_scores.append(score)

        # Reciprocal Rank Fusion (RRF)
        bm25_order = np.argsort(bm25_scores)[::-1]
        bm25_ranks = {int(doc_idx): rank + 1 for rank, doc_idx in enumerate(bm25_order)}

        dense_order = np.argsort(dense_scores)[::-1]
        dense_ranks = {int(doc_idx): rank + 1 for rank, doc_idx in enumerate(dense_order)}

        scored_chunks = []
        for i, chunk in enumerate(self.chunks):
            r_bm25 = bm25_ranks[i]
            r_dense = dense_ranks[i]
            rrf_score = (bm25_weight / (k + r_bm25)) + (dense_weight / (k + r_dense))

            chunk_text = get_chunk_text(chunk)

            scored_chunks.append({
                "id": chunk.get("id", f"chunk_{i}"),
                "source": get_chunk_source_label(chunk),
                "content": chunk_text,
                "score": round(float(rrf_score), 5),
                "bm25_score": round(float(bm25_scores[i]), 4),
                "dense_score": round(float(dense_scores[i]), 4),
                "chunk_index": chunk.get("chunk_index", i),
            })

        # Sort descending by fused RRF score
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]


# Backwards compatibility alias
VectorRetriever = HybridRetriever