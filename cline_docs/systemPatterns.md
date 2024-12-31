# System Patterns

## Architecture
- FastAPI for web API framework
- ChromaDB for vector storage
- Document processing pipeline pattern
- Repository pattern for ChromaDB interactions

## Key Technical Decisions
1. File Processing
   - YAML file parsing using PyYAML
   - Text chunking with configurable size/overlap
   - Sentence-transformers for embedding generation

2. Data Flow
   - Documents → Chunks → Embeddings → ChromaDB
   - Query → Embedding → ChromaDB Search → Results

3. API Design
   - RESTful endpoints
   - JSON response format
   - Error handling middleware
   - Input validation with Pydantic models

## Code Organization
```
my-rag/
├── app/
│   ├── api/
│   │   └── routes.py
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   ├── models/
│   │   └── schemas.py
│   ├── services/
│   │   ├── document_processor.py
│   │   ├── embeddings.py
│   │   └── vector_store.py
│   └── main.py
├── documents/
├── tests/
└── requirements.txt
