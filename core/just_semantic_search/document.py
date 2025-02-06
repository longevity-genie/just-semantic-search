from pathlib import Path
from typing import Optional, TypeVar
from pydantic import BaseModel, Field, ConfigDict, computed_field
import numpy as np
import yaml
import hashlib

from yaml import Dumper

class BugFixDumper(Dumper):
    def represent_str(self, data):
        return self.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    
class Document(BaseModel):
    text: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    vectors: dict[str, list[float]] = Field(default_factory=dict, alias='_vectors')
    token_count: Optional[int] = Field(default=None)
    source: Optional[str] = Field(default=None)

    fragment_num: int | None = None
    total_fragments: int | None = None
    
    model_config = ConfigDict(
        populate_by_name=True,  # Allows both alias and original name to work
        exclude_none=True,      # Don't include None values in serialization
        json_by_alias=True      # Always use aliases in JSON serialization
    )

 

    @property
    def content(self) -> Optional[str]:
        """Returns the text value"""
        return self.text
    
    @computed_field
    def hash(self) -> Optional[str]:
        """Returns MD5 hash of the text"""
        if self.text is None:
            return None
        return hashlib.md5(self.text.encode('utf-8')).hexdigest()
    
    def with_vector(self, embedder_name: str | None, vector: list[float] | np.ndarray | None):
        """Add a vector to the document
        
        Args:
            embedder_name: Name of the embedder used to generate the vector. If it contains '/',
                          only the last segment will be used (e.g., 'model/name' becomes 'name')
            vector: Vector to add, can be list of floats or numpy array
        """
        
        if embedder_name is None or vector is None:
            return self
        
        # Extract last segment of embedder_name if it contains '/'
        processed_name = embedder_name.split('/')[-1]
        
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        self.vectors[processed_name] = vector

        return self

   
    def save_to_yaml(self, path: Path) -> Path:
        """Save document to a YAML file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            yaml.dump(
                self.model_dump(by_alias=True),
                f,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
                Dumper=BugFixDumper
            )
        return path
    
    @staticmethod
    def metadata_overhead(
        tokenizer,
        **metadata
    ) -> int:
        """
        Calculate the adjusted chunk size accounting for metadata tokens.
        
        Args:
            tokenizer: The tokenizer to use for token counting
            max_chunk_size: Original maximum chunk size
            **metadata: Dictionary containing metadata fields
            
        Returns:
            Adjusted maximum chunk size accounting for metadata
        """
        # Extract metadata fields with None as default
        text = metadata.get('text')
        title = metadata.get('title')
        abstract = metadata.get('abstract')
        references = metadata.get('references')
        source = metadata.get('source')
        
        # Build sample metadata text
        metadata_text = "" if text is None else text
        if title:
            metadata_text += f"TITLE: {title}\n"
        if abstract:
            metadata_text += f"ABSTRACT: {abstract}\n"
        #if references:
        #    metadata_text += f"\n\nREFERENCES: {references}"
        if source:
            metadata_text += f"\n\nSOURCE: {source}"
        metadata_text += "\tFRAGMENT: 999/999\n"  # Account for worst-case fragment notation
        
        # Calculate tokens for metadata
        metadata_tokens = len(tokenizer.tokenize(metadata_text))
        
        # Return adjusted size
        return metadata_tokens

    
IDocument = TypeVar('IDocument', bound=Document)  # Document type that must inherit from Document class

class ArticleDocument(Document):

    """Represents a document or document fragment with its metadata"""
    title: str | None = None
    abstract: str | None = None
    references: str | None = None

    @computed_field
    def content(self) -> Optional[str]:
       return self.to_formatted_string()
    
        
    @computed_field
    def hash(self) -> Optional[str]:
        """Returns MD5 hash of the text"""
        if self.text is None:
            return None
        return hashlib.md5(self.to_formatted_string().encode('utf-8')).hexdigest()
   
    

    def to_formatted_string(self, mention_splits: bool = True) -> str:
        """
        Convert the document to a formatted string representation.
        
        Returns:
            Formatted string with metadata and content
        """
        parts = []
        
        if self.title:
            parts.append(f"TITLE: {self.title}\n")
        if self.abstract:
            parts.append(f"ABSTRACT: {self.abstract}\n")
            
        has_multiple_fragments = (self.total_fragments or 0) > 1
        if has_multiple_fragments:
            parts.append("TEXT_FRAGMENT: \n\n")
        
        parts.append(self.text)
        #if self.references:
        #    parts.append(f"\n\nREFERENCES: {self.references}")

        parts.append(f"\n\nSOURCE: {self.source}")
        
        if mention_splits and has_multiple_fragments:
            parts.append(f"\tFRAGMENT: {self.fragment_num}/{self.total_fragments}")
            
        
        parts.append("\n")
        
        # Filter out None values and ensure all parts are strings
        parts = [str(part) for part in parts if part is not None]
        
        return "\n".join(parts)
    