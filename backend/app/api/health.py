from fastapi import APIRouter, Request
from app.schemas.chat import HealthStatus
from app.core.config import settings

router = APIRouter()

@router.get("/health", response_model=HealthStatus)
def health_check(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    chunk_count = pipeline.get_chunk_count() if pipeline else 0
    openai_ok = bool(settings.OPENAI_API_KEY and not settings.OPENAI_API_KEY.startswith("your_"))
    
    return HealthStatus(
        status="healthy",
        version=settings.VERSION,
        chunks_indexed=chunk_count,
        openai_configured=openai_ok
    )
