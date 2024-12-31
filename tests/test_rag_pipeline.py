import pytest
from pathlib import Path
import numpy as np
import logging
from typing import Dict, List, Any

from app.services.document_processor import DocumentProcessor
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.core.config import settings

test_cases = [
    {
        "name": "basic_document_processing",
        "input_query": "peaceful cat",
        "expected_min_chunks": 1,
        "expected_min_results": 1,
        "expected_content_keywords": ["peaceful", "cat"]
    },
    {
        "name": "multiple_results_query",
        "input_query": "cat",
        "expected_min_chunks": 3,
        "expected_min_results": 2,
        "expected_content_keywords": ["cat"]
    }
]

# Print test data for debugging
for case in test_cases:
    logging.info(f"Test case: {case['name']}")
    logging.info(f"Keywords: {case['expected_content_keywords']}")

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestRAGPipeline:
    @pytest.fixture(scope="class")
    def document_processor(self):
        return DocumentProcessor()
    
    @pytest.fixture(scope="class")
    def embedding_service(self):
        return EmbeddingService()
    
    @pytest.fixture(scope="class")
    def vector_store(self):
        store = VectorStore()
        store.reset()  # Clear any existing data
        return store
    
    @pytest.fixture(scope="class")
    def processed_chunks(self, document_processor) -> Dict[str, List[Dict[str, Any]]]:
        return document_processor.process_directory()
    
    @pytest.fixture(scope="class")
    def document_embeddings(self, embedding_service, processed_chunks) -> Dict[str, List[np.ndarray]]:
        embeddings = {}
        for source, docs in processed_chunks.items():
            embeddings[source] = embedding_service.generate_embeddings(docs)
        return embeddings
    
    @pytest.fixture(scope="class")
    def populated_vector_store(self, vector_store, processed_chunks, document_embeddings):
        vector_store.add_documents(processed_chunks, document_embeddings)
        return vector_store

    def test_document_processing(self, processed_chunks):
        """Test that documents are properly processed into structured data."""
        assert processed_chunks, "No documents were generated"
        assert len(processed_chunks) > 0, "No documents were processed"
        
        total_docs = sum(len(docs) for docs in processed_chunks.values())
        assert total_docs > 0, "No documents were generated"
        
        # Verify document structure
        for source, docs in processed_chunks.items():
            for doc in docs:
                assert isinstance(doc, dict), "Each document should be a dictionary"
                assert "type" in doc, "Document should have a type"
                assert "data" in doc, "Document should have data"
                assert all(key in doc["data"] for key in ["title", "description", "link"]), \
                    "Document data should have title, description, and link"

    def test_embedding_generation(self, document_embeddings):
        """Test that embeddings are properly generated for chunks."""
        assert document_embeddings, "No embeddings were generated"
        assert len(document_embeddings) > 0, "No document embeddings were generated"
        
        # Verify embedding dimensions
        for source, embeddings in document_embeddings.items():
            assert len(embeddings) > 0, f"No embeddings generated for {source}"
            assert all(isinstance(emb, np.ndarray) for emb in embeddings), "All embeddings should be numpy arrays"
            assert all(emb.shape == embeddings[0].shape for emb in embeddings), "All embeddings should have same dimensions"

    @pytest.mark.parametrize("test_case", test_cases, ids=lambda tc: tc["name"])
    def test_query_pipeline(self, populated_vector_store, embedding_service, test_case):
        """Test the complete RAG query pipeline with different test cases."""
        # Generate query embedding
        query_embedding = embedding_service.generate_query_embedding(test_case["input_query"])
        assert isinstance(query_embedding, np.ndarray), "Query embedding should be a numpy array"
        
        # Query vector store
        results = populated_vector_store.query(
            query_embedding=query_embedding,
            limit=settings.MAX_RESULTS
        )
        
        # Verify results
        assert len(results) >= test_case["expected_min_results"], \
            f"Expected at least {test_case['expected_min_results']} results"
        
        # Check content relevance
        for result in results:
            assert isinstance(result["score"], float), "Result score should be a float"
            assert 0 <= result["score"] <= 1, "Score should be between 0 and 1"
            assert "type" in result, "Result should have a type"
            assert "data" in result, "Result should have data"
            
            # Check if result contains expected keywords
            content_lower = f"{result['data']['title']} {result['data']['description']}".lower()
            keywords = test_case["expected_content_keywords"]
            logging.info(f"Checking content: {content_lower}")
            logging.info(f"Against keywords: {keywords}")
            found = any(k.lower() in content_lower for k in keywords)
            assert found, \
                f"Result '{content_lower}' should contain at least one of the keywords: {keywords}"

    def test_vector_store_operations(self, vector_store, processed_chunks, document_embeddings):
        """Test vector store operations."""
        # Test reset
        vector_store.reset()
        
        # Test adding documents
        vector_store.add_documents(processed_chunks, document_embeddings)
        
        # Test querying after reset and re-add
        query_embedding = np.random.rand(384)  # Match the embedding dimension
        results = vector_store.query(query_embedding)
        
        assert isinstance(results, list), "Query results should be a list"
        assert len(results) > 0, "Should get at least one result"
        assert all(isinstance(r, dict) for r in results), "All results should be dictionaries"
        assert all({"source", "score", "type", "data"} <= r.keys() for r in results), \
            "Results should have source, score, type, and data"
        assert all({"title", "description", "link"} <= r["data"].keys() for r in results), \
            "Result data should have title, description, and link"
