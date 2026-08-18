from typing import List, Optional
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the sender ('user' or 'assistant')")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User question or prompt")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Previous conversation turns")
    top_k: Optional[int] = Field(default=3, description="Number of source chunks to retrieve")

class SourceDocument(BaseModel):
    id: str = Field(..., description="Unique chunk ID")
    source: str = Field(..., description="Source document file name")
    content: str = Field(..., description="Retrieved chunk text content")
    score: float = Field(..., description="Similarity match score (0.0 - 1.0)")
    chunk_index: int = Field(..., description="Index of chunk within source file")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Synthesized AI response based on retrieved knowledge")
    sources: List[SourceDocument] = Field(default=[], description="Source citations used to compose answer")

class HealthStatus(BaseModel):
    status: str
    version: str
    chunks_indexed: int
    openai_configured: bool
