from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    APP_NAME: str = "RAG API"
    APP_VERSION: str = "1.0.0"
    DOCUMENTS_DIR: str = "documents"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHROMADB_DIR: str = ".chromadb"
    MAX_RESULTS: int = 5
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    class Config:
        env_file = ".env"

settings = Settings()
