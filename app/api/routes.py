from fastapi import APIRouter, HTTPException
from pathlib import Path
import logging
from typing import Dict, List

from app.models.schemas import QueryRequest, QueryResponse, ProcessingStatus
from app.services.document_processor import DocumentProcessor
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
document_processor = DocumentProcessor()
embedding_service = EmbeddingService()
vector_store = VectorStore()

@router.post("/process", response_model=ProcessingStatus)
async def process_documents():
    """Process all documents in the documents directory and store them in the vector store."""
    try:
        # Process documents into chunks
        chunks: Dict[str, List[str]] = document_processor.process_directory()
        
        if not chunks:
            raise HTTPException(status_code=404, detail="No documents found in the documents directory")
        
        # Generate embeddings for each document's chunks
        embeddings = {}
        for source, source_chunks in chunks.items():
            embeddings[source] = embedding_service.generate_embeddings(source_chunks)
        
        # Reset vector store and add new documents
        vector_store.reset()
        vector_store.add_documents(chunks, embeddings)
        
        total_chunks = sum(len(chunk_list) for chunk_list in chunks.values())
        
        return ProcessingStatus(
            total_documents=len(chunks),
            total_chunks=total_chunks,
            status="Documents processed and indexed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the vector store for relevant document chunks."""
    try:
        # Generate embedding for query
        query_embedding = embedding_service.generate_query_embedding(request.query)
        
        # Query vector store
        results = vector_store.query(
            query_embedding=query_embedding,
            limit=request.limit or settings.MAX_RESULTS
        )
        
        return QueryResponse(
            query=request.query,
            results=results,
            total_chunks=len(results)
        )
        
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
