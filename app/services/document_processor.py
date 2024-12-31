import yaml
from pathlib import Path
from typing import List, Dict, Any, Union
import logging
from app.core.config import settings

class DocumentProcessor:
    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, 
                 chunk_overlap: int = settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)

    def read_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Read and parse a YAML file."""
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error reading YAML file {file_path}: {str(e)}")
            raise

    def get_document_type(self, file_path: Path) -> str:
        """Determine document type from filename."""
        filename = file_path.stem.lower()
        if 'product' in filename:
            return 'product'
        elif 'page' in filename:
            return 'page'
        else:
            raise ValueError(f"Unknown document type for file: {file_path}")

    def process_yaml_content(self, content: Dict[str, Any], doc_type: str) -> List[Dict[str, Any]]:
        """Process YAML content into structured documents."""
        documents = []
        
        if not isinstance(content, list):
            self.logger.warning("YAML content is not a list of documents")
            return documents
            
        for item in content:
            doc = None
            if doc_type == 'product' and 'Product' in item:
                doc = item['Product']
            elif doc_type == 'page' and 'Page' in item:
                doc = item['Page']
                
            if doc:
                # Clean and normalize text fields
                title = doc.get('title', '').strip()
                description = doc.get('description', '').strip()
                # Replace newlines with spaces in description
                description = ' '.join(description.split())
                
                documents.append({
                    'type': doc_type,
                    'data': {
                        'title': title,
                        'description': description,
                        'link': doc.get('link', '')
                    }
                })
                self.logger.info(f"Processed {doc_type} document: {title}")
            else:
                self.logger.warning(f"Skipping invalid document: {item}")
        
        return documents

    def process_document(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single document into structured data."""
        try:
            yaml_content = self.read_yaml_file(file_path)
            if not yaml_content:
                self.logger.warning(f"Empty or invalid YAML content in {file_path}")
                return []
                
            doc_type = self.get_document_type(file_path)
            documents = self.process_yaml_content(yaml_content, doc_type)
            
            if not documents:
                self.logger.warning(f"No valid documents extracted from {file_path}")
                return []
            
            self.logger.info(f"Processed {file_path} into {len(documents)} documents")
            return documents
        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {str(e)}")
            raise

    def process_directory(self, directory: Path = Path(settings.DOCUMENTS_DIR)) -> Dict[str, List[Dict[str, Any]]]:
        """Process all YAML files in a directory."""
        processed_documents = {}
        
        try:
            yaml_files = list(directory.glob('*.yml'))
            self.logger.info(f"Found YAML files: {yaml_files}")
            
            # Process products.yml first to ensure cat-related documents are included
            for file_path in sorted(yaml_files, key=lambda x: x.name != 'products.yml'):
                self.logger.info(f"Processing {file_path}")
                documents = self.process_document(file_path)
                if documents:
                    processed_documents[str(file_path)] = documents
                    self.logger.info(f"Added {len(documents)} documents from {file_path}")
            
            if not processed_documents:
                self.logger.warning("No valid documents were processed")
                
            return processed_documents
        except Exception as e:
            self.logger.error(f"Error processing directory {directory}: {str(e)}")
            raise
