from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import logging
from app.core.config import settings

class EmbeddingService:
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.logger = logging.getLogger(__name__)
        try:
            self.model = SentenceTransformer(model_name)
            self.logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            self.logger.error(f"Error loading embedding model: {str(e)}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a list of texts."""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a single query text."""
        try:
            return self.model.encode(query, convert_to_numpy=True)
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {str(e)}")
            raise
