# import os
# import sys
# import json
# import logging
# from pathlib import Path

# # Add backend directory to path
# current_dir = Path(__file__).resolve().parent
# backend_dir = current_dir.parent / "backend"
# sys.path.append(str(backend_dir))

# from app.rag.loader import DocumentLoader
# from app.rag.chunker import DocumentChunker
# from app.rag.embeddings import EmbeddingService
# from app.core.config import settings

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# def run_ingestion(docs_dir: Path = None, output_file: Path = None):
#     docs_dir = Path(docs_dir or settings.DOCUMENTS_PATH)
#     output_file = Path(output_file or settings.CHUNKS_PATH)
    
#     logger.info(f"Starting document ingestion from: {docs_dir.resolve()}")
    
#     if not docs_dir.exists():
#         logger.error(f"Documents directory does not exist: {docs_dir}")
#         return

#     # 1. Load documents
#     loader = DocumentLoader()
#     documents = loader.load_directory(docs_dir)
#     logger.info(f"Loaded {len(documents)} document files.")

#     # 2. Chunk documents
#     chunker = DocumentChunker(chunk_size=150, chunk_overlap=30)
#     chunks = chunker.chunk_documents(documents)
#     logger.info(f"Generated {len(chunks)} text chunks.")

#     # 3. Compute Embeddings
#     embedding_service = EmbeddingService(
#         model_name=settings.EMBEDDING_MODEL,
#         api_key=settings.OPENAI_API_KEY
#     )
    
#     logger.info("Computing embeddings for chunks...")
#     for idx, chunk in enumerate(chunks, 1):
#         chunk["embedding"] = embedding_service.get_embedding(chunk["content"])
#         if idx % 5 == 0 or idx == len(chunks):
#             logger.info(f"Processed embeddings {idx}/{len(chunks)}")

#     # 4. Save to JSON
#     output_file.parent.mkdir(parents=True, exist_ok=True)
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(chunks, f, indent=2, ensure_ascii=False)

#     logger.info(f"Ingestion complete! Knowledge base saved to {output_file.resolve()}")

# if __name__ == "__main__":
#     run_ingestion()
