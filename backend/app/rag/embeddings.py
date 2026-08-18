import logging
import hashlib
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import settings


logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, embedding_model: str = None):
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self.st_model = None

        if self.embedding_model:
            try:
                logger.info(f"Loading SentenceTransformer model: {self.embedding_model}")
                self.st_model = SentenceTransformer(self.embedding_model)
                logger.info(f"Successfully initialized {self.embedding_model}")
            except Exception as e:
                logger.warning(
                    f"Could not load SentenceTransformer '{self.embedding_model}': {e}. "
                    "Falling back to deterministic hashing-based vectors."
                )
        else:
            logger.info("No embedding_model specified. Using deterministic hashing-based fallback vectors.")

    def get_embedding(self, text: str) -> List[float]:
        """Gets dense vector embedding representation for text."""
        if self.st_model is not None:
            try:
                vec = self.st_model.encode(text, normalize_embeddings=True)
                return vec.tolist()
            except Exception as e:
                logger.warning(f"SentenceTransformer encoding failed: {e}")

        return self._fallback_embedding(text)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding computation."""
        if self.st_model is not None:
            try:
                vecs = self.st_model.encode(texts, normalize_embeddings=True)
                return vecs.tolist()
            except Exception as e:
                logger.warning(f"SentenceTransformer batch encoding failed: {e}")

        return [self.get_embedding(text) for text in texts]

    def _fallback_embedding(self, text: str, dim: int = 1024) -> List[float]:
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
