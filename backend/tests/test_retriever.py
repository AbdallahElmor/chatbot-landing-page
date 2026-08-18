import pytest
from app.rag.embeddings import EmbeddingService
from app.rag.retriever import HybridRetriever, VectorRetriever

def test_retriever_scoring():
    embeddings = EmbeddingService()
    retriever = HybridRetriever(embedding_service=embeddings)
    
    sample_chunks = [
        {
            "id": "chunk_1",
            "source": "pricing.md",
            "content": "Our enterprise subscription tier costs $99 per month per seat.",
            "chunk_index": 0
        },
        {
            "id": "chunk_2",
            "source": "contact.md",
            "content": "You can reach customer support at support@company.com or call 1-800-555-0199.",
            "chunk_index": 0
        }
    ]
    
    retriever.add_chunks(sample_chunks)
    
    results = retriever.retrieve(query="How much does enterprise plan cost?", top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "chunk_1"
    assert results[0]["score"] > 0.0
    assert "bm25_score" in results[0]
    assert "dense_score" in results[0]

def test_hybrid_bm25_and_dense():
    embeddings = EmbeddingService()
    retriever = HybridRetriever(embedding_service=embeddings)

    sample_chunks = [
        {
            "id": "faq_timeline",
            "source": "faqs.json",
            "content": "Topic: project-timeline\nQuestion (EN): How long does a typical engagement take?\nAnswer (EN): MVP builds ship in 6-10 weeks.",
            "chunk_index": 0
        },
        {
            "id": "faq_location",
            "source": "faqs.json",
            "content": "Topic: location\nQuestion (EN): Where are your offices located?\nAnswer (EN): We are located in Egypt and GCC.",
            "chunk_index": 1
        }
    ]

    retriever.set_chunks(sample_chunks)
    results = retriever.retrieve(query="6-10 weeks MVP timeline", top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "faq_timeline"
