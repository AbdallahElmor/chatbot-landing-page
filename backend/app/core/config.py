import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolved once at import time — works both locally and in Docker
_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_PROJECT_ROOT = _BASE_DIR.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "Synkro AI API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    DEBUG: bool = True # in development, set to True for hot-reloading
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # AI & Embeddings
    GROQ_API_KEY: Optional[str] = None        # Groq API key (used for LLM generation)
    OPENAI_API_KEY: Optional[str] = None      # Kept for compatibility
    EMBEDDING_MODEL: Optional[str] = "gemini-embedding-001"
    GEMINI_API_KEY: Optional[str] = None

    OPENAI_MODEL: Optional[str] = "openai/gpt-oss-20b"
    GEMINI_MODEL: Optional[str] = "gemini-2.5-flash"
    

    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CHUNKS_PATH: Path = DATA_DIR / "Synkro-data.json"

    model_config = SettingsConfigDict(
        # Resolve .env relative to this file: backend/app/core/ -> go up 4 levels to project root
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
