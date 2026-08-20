import logging
import hashlib
from typing import List
import numpy as np
from google import genai
from app.core.config import settings


logger = logging.getLogger(__name__)

# Gemini embed_content accepts at most 100 texts per call
_BATCH_LIMIT = 100


class EmbeddingService:
    """Embedding service using Gemini text-embedding API."""

    def __init__(self, embedding_model: str = None):
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self.client = None

        if settings.GEMINI_API_KEY:
            try:
                self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
                logger.info(f"✅ Gemini embedding client initialized | model={self.embedding_model}")
            except Exception as e:
                logger.warning(
                    f"Could not initialize Gemini embedding client: {e}. "
                    "Falling back to deterministic hashing-based vectors."
                )
        else:
            logger.warning(
                "No GEMINI_API_KEY set. Using deterministic hashing-based fallback vectors."
            )

    def get_embedding(self, text: str) -> List[float]:
        """Gets dense vector embedding representation for text."""
        if self.client is not None:
            try:
                result = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=[text],
                )
                return list(result.embeddings[0].values)
            except Exception as e:
                logger.warning(f"Gemini embedding failed: {e}")

        return self._fallback_embedding(text)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding computation."""
        if self.client is not None:
            try:
                all_embeddings = []
                # Process in batches of _BATCH_LIMIT
                for i in range(0, len(texts), _BATCH_LIMIT):
                    batch = texts[i : i + _BATCH_LIMIT]
                    result = self.client.models.embed_content(
                        model=self.embedding_model,
                        contents=batch,
                    )
                    all_embeddings.extend(
                        list(emb.values) for emb in result.embeddings
                    )
                return all_embeddings
            except Exception as e:
                logger.warning(f"Gemini batch embedding failed: {e}")

        return [self._fallback_embedding(text) for text in texts]

    def _fallback_embedding(self, text: str, dim: int = 768) -> List[float]:
        """Deterministic hashing-based embedding vector generator for fallback/testing."""
        words = text.lower().split()
        vector = np.zeros(dim)
        for word in words:
            h = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
            vector[h] += 1.0

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        else:
            vector = np.ones(dim) / np.sqrt(dim)
        return vector.tolist()
