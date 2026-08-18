import logging
from pathlib import Path
from typing import Dict, Any, List
import json
from app.rag.embeddings import EmbeddingService
from app.rag.retriever import HybridRetriever
from app.rag.generator import ResponseGenerator
from app.core.config import settings

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Unified RAG Pipeline combining Hybrid (BM25 + BAAI/bge-m3) retrieval directly from the data folder and LLM generation."""

    def __init__(self, data_dir: Path = None, chunks_path: Path = None):
        self.embeddings = EmbeddingService(
            embedding_model=settings.EMBEDDING_MODEL
        )
        self.retriever = HybridRetriever(embedding_service=self.embeddings)
        self.generator = ResponseGenerator(
            openai_model=settings.OPENAI_MODEL,
            gemini_model=settings.GEMINI_MODEL,
            groq_api_key=settings.GROQ_API_KEY,
            gemini_api_key=settings.GEMINI_API_KEY,
        )
        chunks_file = chunks_path or settings.CHUNKS_PATH
        chunks = self._load_chunks(chunks_file)
        self.retriever.set_chunks(chunks)
        logger.info(f"Loaded {len(chunks)} pre-chunked entries from {chunks_file}")

    def _load_chunks(self, path: str) -> List[Dict[str, Any]]:
        """Loads the already-chunked knowledge base JSON directly from disk.
        No document loading or chunking is performed here — the JSON file
        is expected to already be structured as a list of chunk objects."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Chunks file not found at '{path}'.")

        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Chunks file at '{path}' is not valid JSON: {e}")

        if not isinstance(data, list):
            raise ValueError(
                f"Expected a list of chunk objects in '{path}', got {type(data).__name__}."
            )

        return data

    def query(self, message: str, history: List[Dict[str, str]] = None, top_k: int = 3) -> Dict[str, Any]:
        """Runs the query through hybrid retrieval and context generation."""
        try:
            retrieved_sources = self.retriever.retrieve(query=message, top_k=top_k)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            retrieved_sources = []

        answer = self.generator.generate_response(
            query=message,
            context_chunks=retrieved_sources,
            history=history
        )

        return {
            "answer": answer,
            "sources": retrieved_sources
        }

    def get_chunk_count(self) -> int:
        return len(getattr(self.retriever, "chunks", []))