from pathlib import Path
import re
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from abc import ABC, abstractmethod
import numpy as np
import pydantic_numpy.typing as pnd
import yaml

class Document(BaseModel):
    content: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    vectors: list[float] = Field(default_factory=list, alias="_vectors")  # Changed to match parent class


    def set_vectors(self, vectors: np.ndarray | list[float]) -> None:
        """Set the document's vector embeddings"""
        self.vectors = vectors.tolist() if isinstance(vectors, np.ndarray) else vectors

    def get_vectors(self) -> np.ndarray:
        """Get the document's vector embeddings"""
        return np.array(self.vectors)

    model_config = ConfigDict(
        populate_by_name=True,
        exclude_defaults=True,
        json_encoders={
            np.ndarray: lambda x: x.tolist()
        }
    )

    def save_to_yaml(self, path: Path) -> Path:
        """Save document to a YAML file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            yaml.dump(
                self.model_dump(),
                f,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )
        return path


class ArticleDocument(Document):

    """Represents a document or document fragment with its metadata"""
    title: str | None
    abstract: str | None
    source: str
    fragment_num: int
    total_fragments: int
   
    

    def to_formatted_string(self, mention_splits: bool = True) -> str:
        """
        Convert the document to a formatted string representation.
        
        Args:
            mention_splits: Whether to include fragment information
        
        Returns:
            Formatted string with metadata and content
        """
        parts = []
        
        if self.title:
            parts.append(f"TITLE: {self.title}\n")
        if self.abstract:
            parts.append(f"ABSTRACT: {self.abstract}\n")
            
        has_multiple_fragments = self.total_fragments > 1
        if has_multiple_fragments:
            parts.append("TEXT_FRAGMENT: ")
        
        parts.append(self.content)
        
        parts.append(f"\n\nSOURCE: {self.source}")
        if mention_splits and has_multiple_fragments:
            parts.append(f"\tFRAGMENT: {self.fragment_num}/{self.total_fragments}")
        
        parts.append("\n")
        
        return "\n".join(parts)
    
    def with_extended_content(self, mention_splits: bool = True) -> "ArticleDocument":
        """Create a new document with extended content"""
        if self.content is None or self.source not in self.content:
            self.content = self.to_formatted_string(mention_splits)
        return self

    @staticmethod
    def calculate_adjusted_chunk_size(
        tokenizer,
        max_chunk_size: int,
        title: str | None = None,
        abstract: str | None = None,
        source: str | None = None
    ) -> int:
        """
        Calculate the adjusted chunk size accounting for metadata tokens.
        
        Args:
            tokenizer: The tokenizer to use for token counting
            max_chunk_size: Original maximum chunk size
            title: Optional title text
            abstract: Optional abstract text
            source: Optional source identifier
            
        Returns:
            Adjusted maximum chunk size accounting for metadata
        """
        # Build sample metadata text
        metadata_text = ""
        if title:
            metadata_text += f"TITLE: {title}\n"
        if abstract:
            metadata_text += f"ABSTRACT: {abstract}\n"
        if source:
            metadata_text += f"\n\nSOURCE: {source}"
        metadata_text += "\tFRAGMENT: 999/999\n"  # Account for worst-case fragment notation
        
        # Calculate tokens for metadata
        metadata_tokens = len(tokenizer.tokenize(metadata_text))
        
        # Return adjusted size
        return max_chunk_size - metadata_tokens