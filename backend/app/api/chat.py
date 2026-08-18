from fastapi import APIRouter, HTTPException, Request
from app.schemas.chat import ChatRequest, ChatResponse, SourceDocument
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request_body: ChatRequest, request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    if not pipeline:
        raise HTTPException(status_code=500, detail="RAG Pipeline is not initialized")
    
    try:
        history_dicts = None
        if request_body.history:
            history_dicts = [{"role": m.role, "content": m.content} for m in request_body.history]

        result = pipeline.query(
            message=request_body.message,
            history=history_dicts,
            top_k=request_body.top_k or 3
        )
        
        # formatted_sources = [
        #     SourceDocument(
        #         id=s.get("id", "chunk"),
        #         source=s.get("source", "unknown"),
        #         content=s.get("content", ""),
        #         score=s.get("score", 0.0),
        #         chunk_index=s.get("chunk_index", 0)
        #     )
        #     for s in result.get("sources", [])
        # ]
        
        return ChatResponse(
            answer=result.get("answer", "No answer generated."),
            # sources=formatted_sources
        )
    except Exception as e:
        logger.error(f"Error handling chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
