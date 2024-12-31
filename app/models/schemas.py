from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The search query text")
    limit: Optional[int] = Field(5, ge=1, le=20, description="Maximum number of results to return")

class DocumentChunk(BaseModel):
    content: str = Field(..., description="The content of the document chunk")
    source: str = Field(..., description="The source document path")
    score: float = Field(..., ge=0, le=1, description="Relevance score")

class QueryResponse(BaseModel):
    query: str = Field(..., description="The original search query")
    results: List[DocumentChunk] = Field(..., description="List of relevant document chunks")
    total_chunks: int = Field(..., ge=0, description="Total number of chunks found")

class ProcessingStatus(BaseModel):
    total_documents: int = Field(..., ge=0, description="Total number of documents processed")
    total_chunks: int = Field(..., ge=0, description="Total number of chunks created")
    status: str = Field(..., description="Processing status message")
