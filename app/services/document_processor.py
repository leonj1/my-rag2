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

    def flatten_dict(self, d: Union[Dict[str, Any], List[Any]], parent_key: str = '') -> List[str]:
        """Flatten a nested dictionary or list into a list of strings."""
        texts = []
        
        if isinstance(d, list):
            # For top-level list, process each item
            for item in d:
                if isinstance(item, dict) and 'Product' in item:
                    product = item['Product']
                    # Create a meaningful text representation of the product
                    product_text = f"Title: {product.get('title', '')}\n"
                    product_text += f"Description: {product.get('description', '').strip()}\n"
                    product_text += f"Link: {product.get('link', '')}"
                    texts.append(product_text)
        elif isinstance(d, dict):
            # For nested dictionaries
            for key, value in d.items():
                if isinstance(value, dict):
                    texts.extend(self.flatten_dict(value))
                elif isinstance(value, list):
                    texts.extend(self.flatten_dict(value))
                else:
                    texts.append(f"{key}: {str(value)}")
        
        return texts

    def create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # If not at the end, try to break at a newline
            if end < len(text):
                last_newline = chunk.rfind('\n')
                if last_newline != -1:
                    end = start + last_newline + 1
                    chunk = text[start:end]
            
            chunks.append(chunk)
            start = end - self.chunk_overlap

        return chunks

    def process_document(self, file_path: Path) -> List[str]:
        """Process a single document into chunks."""
        yaml_content = self.read_yaml_file(file_path)
        if not yaml_content:  # Handle empty files
            self.logger.warning(f"Empty or invalid YAML content in {file_path}")
            return []
            
        flattened_texts = self.flatten_dict(yaml_content)
        
        # Filter out empty strings
        filtered_texts = [text for text in flattened_texts if text.strip()]
        if not filtered_texts:  # Handle case where no valid text was extracted
            self.logger.warning(f"No valid text content extracted from {file_path}")
            return []
        
        # Each text block becomes its own chunk
        chunks = []
        for text in filtered_texts:
            # Only create sub-chunks if text exceeds chunk size
            if len(text) > self.chunk_size:
                chunks.extend(self.create_chunks(text))
            else:
                chunks.append(text)
        
        # Log for debugging
        self.logger.info(f"Processed document {file_path} into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            self.logger.info(f"Chunk {i}: {chunk[:100]}...")
        
        return chunks

    def process_directory(self, directory: Path = Path(settings.DOCUMENTS_DIR)) -> Dict[str, List[str]]:
        """Process all YAML files in a directory."""
        document_chunks = {}
        
        try:
            yaml_files = list(directory.glob('*.yml'))
            for file_path in yaml_files:
                self.logger.info(f"Processing {file_path}")
                chunks = self.process_document(file_path)
                if chunks:  # Only include documents that produced valid chunks
                    document_chunks[str(file_path)] = chunks
            
            if not document_chunks:
                self.logger.warning("No valid documents were processed")
                
            return document_chunks
        except Exception as e:
            self.logger.error(f"Error processing directory {directory}: {str(e)}")
            raise
