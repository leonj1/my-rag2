from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
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

    def generate_embeddings(self, documents: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Generate embeddings for a list of documents."""
        try:
            # Extract text content from documents for embedding
            texts = []
            for doc in documents:
                title = doc['data']['title']
                description = doc['data']['description']
                # Create text with title emphasis and proper spacing
                text = f"{title} {title} {title}. {description}"
                texts.append(text)
                self.logger.info(f"Processing text: {text}")
            
            # Generate embeddings with normalization
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalize embeddings
                batch_size=1  # Process one at a time for better quality
            )
            self.logger.info(f"Generated embeddings for {len(texts)} documents")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a single query text."""
        try:
            # Generate query embedding with normalization
            query_embedding = self.model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalize embeddings
                show_progress_bar=False
            )
            self.logger.info(f"Generated embedding for query: {query}")
            return query_embedding
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {str(e)}")
            raise
