import pytest
from app.rag.chunker import DocumentChunker

def test_chunker_basic():
    chunker = DocumentChunker(chunk_size=10, chunk_overlap=2)
    sample_doc = {
        "source": "test.txt",
        "content": "Word " * 25
    }
    
    chunks = chunker.chunk_document(sample_doc)
    assert len(chunks) > 1
    assert chunks[0]["source"] == "test.txt"
    assert chunks[0]["chunk_index"] == 0
    assert "content" in chunks[0]

def test_chunker_empty():
    chunker = DocumentChunker(chunk_size=10, chunk_overlap=2)
    sample_doc = {
        "source": "empty.txt",
        "content": ""
    }
    chunks = chunker.chunk_document(sample_doc)
    assert len(chunks) == 0
