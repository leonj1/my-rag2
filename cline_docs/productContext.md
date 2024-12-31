# Product Context

## Purpose
This project implements a Retrieval-Augmented Generation (RAG) system that enables efficient document search and retrieval using vector embeddings.

## Problems Solved
- Efficient document search and retrieval from a collection of YAML files
- Semantic search capabilities through vector embeddings
- Structured API access to document retrieval functionality

## Expected Functionality
1. Document Processing
   - Read YAML files from the documents folder
   - Chunk documents into appropriate segments
   - Generate embeddings for each chunk

2. Storage
   - Store document chunks and embeddings in ChromaDB
   - Maintain relationships between chunks and source documents

3. Retrieval
   - Provide API endpoint for semantic search
   - Return relevant document chunks based on query
   - Include source document references
