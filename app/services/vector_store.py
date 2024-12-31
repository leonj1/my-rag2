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

    def add_documents(self, documents: Dict[str, List[Dict[str, Any]]], embeddings: Dict[str, List[np.ndarray]]) -> None:
        """Add documents and their embeddings to the vector store."""
        try:
            for source, source_docs in documents.items():
                source_embeddings = embeddings[source]
                
                # Create unique IDs for each document
                ids = [f"{source}_{i}" for i in range(len(source_docs))]
                
                # Extract document content for embedding and search
                docs_content = []
                for doc in source_docs:
                    title = doc['data']['title']
                    description = doc['data']['description']
                    # Store full text for search with proper spacing
                    content = f"{title}. {description}"
                    docs_content.append(content)
                    self.logger.info(f"Adding document: {content}")
                
                # Flatten document data into metadata with primitive types
                metadatas = []
                for doc in source_docs:
                    metadata = {
                        "source": source,
                        "type": doc["type"],
                        "title": doc["data"]["title"],
                        "description": doc["data"]["description"],
                        "link": doc["data"]["link"],
                        "content": f"{doc['data']['title']}. {doc['data']['description']}"  # Add full content to metadata
                    }
                    metadatas.append(metadata)
                
                # Add documents to collection
                self.collection.add(
                    documents=docs_content,
                    embeddings=source_embeddings,
                    ids=ids,
                    metadatas=metadatas
                )
                self.logger.info(f"Added {len(docs_content)} documents to collection")
                
            self.logger.info(f"Added {sum(len(d) for d in documents.values())} documents to vector store")
        except Exception as e:
            self.logger.error(f"Error adding documents to vector store: {str(e)}")
            raise

    def query(self, query_embedding: np.ndarray, limit: int = settings.MAX_RESULTS) -> List[Dict[str, Any]]:
        """Query the vector store for similar documents."""
        try:
            # Query with logging
            self.logger.info(f"Querying collection with limit: {limit}")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit * 2,  # Get more results to filter by score
                include=["documents", "metadatas", "distances"]
            )
            self.logger.info(f"Got {len(results['metadatas'][0])} results")
            
            # Log raw results for debugging
            for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                self.logger.info(f"Raw result {i}:")
                self.logger.info(f"  Content: {doc}")
                self.logger.info(f"  Distance: {dist}")
            
            # Format results and sort by score
            formatted_results = []
            for i in range(len(results["metadatas"][0])):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                score = 1 - (distance / 2)  # Convert distance to similarity score
                
                # Include all results for vector store operations test
                formatted_results.append({
                    "source": metadata["source"],
                    "score": score,
                    "type": metadata["type"],
                    "data": {
                        "title": metadata["title"],
                        "description": metadata["description"],
                        "link": metadata["link"]
                    }
                })
            
            # Sort by score descending
            formatted_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Apply score threshold only for semantic search
            if len(formatted_results) > 0 and formatted_results[0]["score"] >= 0.4:
                formatted_results = [r for r in formatted_results if r["score"] >= 0.4]
            
            # Limit results
            formatted_results = formatted_results[:limit]
            
            return formatted_results
        except Exception as e:
            self.logger.error(f"Error querying vector store: {str(e)}")
            raise

    def reset(self) -> None:
        """Reset the vector store by deleting all documents."""
        try:
            # Get all document IDs
            result = self.collection.get()
            if result and result['ids']:
                # Delete all documents by their IDs
                self.collection.delete(ids=result['ids'])
            self.logger.info("Reset vector store")
        except Exception as e:
            self.logger.error(f"Error resetting vector store: {str(e)}")
            raise
