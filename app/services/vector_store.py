import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
import numpy as np
from typing import List, Dict, Any
import logging
from pathlib import Path
from app.core.config import settings

class VectorStore:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = chromadb.PersistentClient(
            path=settings.CHROMADB_DIR,
            settings=ChromaSettings(
                allow_reset=True,
                is_persistent=True
            )
        )
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.EMBEDDING_MODEL
            )
        )
        self.logger.info("Initialized ChromaDB vector store")

    def add_documents(self, chunks: Dict[str, List[str]], embeddings: Dict[str, List[np.ndarray]]) -> None:
        """Add document chunks and their embeddings to the vector store."""
        try:
            for source, source_chunks in chunks.items():
                source_embeddings = embeddings[source]
                
                # Create unique IDs for each chunk
                ids = [f"{source}_{i}" for i in range(len(source_chunks))]
                
                # Add chunks to collection
                self.collection.add(
                    documents=source_chunks,
                    embeddings=source_embeddings,
                    ids=ids,
                    metadatas=[{"source": source} for _ in source_chunks]
                )
                
            self.logger.info(f"Added {sum(len(c) for c in chunks.values())} chunks to vector store")
        except Exception as e:
            self.logger.error(f"Error adding documents to vector store: {str(e)}")
            raise

    def query(self, query_embedding: np.ndarray, limit: int = settings.MAX_RESULTS) -> List[Dict[str, Any]]:
        """Query the vector store for similar chunks."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "content": results["documents"][0][i],
                    "source": results["metadatas"][0][i]["source"],
                    "score": 1 - (results["distances"][0][i] / 2)  # Convert distance to similarity score
                })
            
            return formatted_results
        except Exception as e:
            self.logger.error(f"Error querying vector store: {str(e)}")
            raise

    def reset(self) -> None:
        """Reset the vector store by deleting all documents."""
        try:
            self.collection.delete(where={})
            self.logger.info("Reset vector store")
        except Exception as e:
            self.logger.error(f"Error resetting vector store: {str(e)}")
            raise
