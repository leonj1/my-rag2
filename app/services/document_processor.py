import yaml
from pathlib import Path
from typing import List, Dict, Any
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

    def flatten_dict(self, d: Dict[str, Any], parent_key: str = '') -> List[str]:
        """Flatten a nested dictionary into a list of strings."""
        texts = []
        for key, value in d.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            
            if isinstance(value, dict):
                texts.extend(self.flatten_dict(value, new_key))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        texts.extend(self.flatten_dict(item, f"{new_key}[{i}]"))
                    else:
                        texts.append(f"{new_key}[{i}]: {str(item)}")
            else:
                texts.append(f"{new_key}: {str(value)}")
        
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
        flattened_text = self.flatten_dict(yaml_content)
        
        # Join flattened text with newlines
        full_text = '\n'.join(flattened_text)
        
        return self.create_chunks(full_text)

    def process_directory(self, directory: Path = Path(settings.DOCUMENTS_DIR)) -> Dict[str, List[str]]:
        """Process all YAML files in a directory."""
        document_chunks = {}
        
        try:
            yaml_files = list(directory.glob('*.yml'))
            for file_path in yaml_files:
                self.logger.info(f"Processing {file_path}")
                document_chunks[str(file_path)] = self.process_document(file_path)
                
            return document_chunks
        except Exception as e:
            self.logger.error(f"Error processing directory {directory}: {str(e)}")
            raise
