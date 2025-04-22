from abc import ABC
import json
from just_semantic_search.splitters.abstract_splitters import AbstractSplitter, SentenceTransformerMixin
from typing import List, TypeAlias, Generic, Optional
import numpy as np
from pathlib import Path
import re
from just_semantic_search.document import ArticleDocument, Document, IDocument
from pydantic import Field

from just_semantic_search.document import Document, IDocument
from typing import Generic, List, Optional, TypeAlias
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
from just_semantic_search.remote.jina import JinaEmbeddingData

class AbstractDictionarySplitter(AbstractSplitter[dict, IDocument], ABC):
    """Implementation of AbstractSplitter for text content that works with any Document type."""

    content_key: str = "content"
    extend_content: bool = True
    
    def compute_metadata_header(self, metadata: dict, source: str | None = None, fragment_num: int = None, total_fragments: int = None) -> str:
        """
        Creates a markdown header with metadata fields and fragment information
        
        Args:
            metadata: Dictionary of metadata fields
            source: Optional source information
            fragment_num: Current fragment number
            total_fragments: Total number of fragments
            
        Returns:
            Text with metadata formatted as markdown
        """
        if not metadata and not source and fragment_num is None and not self.extend_content:
            return ""
            
        # Start with a title if available
        header = ""
        if "title" in metadata:
            header += f"# {metadata['title']}\n\n"
            
        # Add fragment information if available
        if fragment_num is not None and total_fragments is not None:
            header += f"**Fragment:** {fragment_num} of {total_fragments}  \n"
            
        # Add other metadata fields
        for key, value in metadata.items():
            if key != "title" and value:  # Skip empty values
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
                elif not isinstance(value, (str, int, float, bool)):
                    continue  # Skip complex objects
                
                header += f"**{key.replace('_', ' ').title()}:** {value}  \n"
        if source is not None:
            header += f"**Source:** {source}  \n"
            
        # Add a content header instead of a visual separator
        if header:
            header += "\n## Content\n\n"
            
        return header
        
  

    def split(self, text: dict, embed: bool = True, source: str | None = None, metadata: Optional[dict] = None, **kwargs) -> List[IDocument]:
    
        # Extract all fields except content_key as metadata
        extracted_metadata = {k: v for k, v in text.items() if k != self.content_key}
        
        # Merge with optional metadata if provided
        if metadata:
            extracted_metadata.update(metadata)
            
        content = text[self.content_key]
        
        # Calculate a sample metadata header to estimate overhead
        sample_header = self.compute_metadata_header(extracted_metadata, source, 1, 1) if self.extend_content else ""
        metadata_overhead = len(self.tokenizer.tokenize(sample_header)) if sample_header else 0
        
        # Get tokens and chunks with consideration for metadata overhead
        token_chunks, text_chunks = self.get_tokens_and_chunks(content, metadata_overhead=metadata_overhead)
        total_fragments = len(text_chunks)
        
        # Calculate metadata headers for each fragment with proper fragment information
        if self.extend_content:
            text_chunks = [
                self.compute_metadata_header(extracted_metadata, source, i+1, total_fragments) + chunk 
                for i, chunk in enumerate(text_chunks)
            ]
        
        # Generate embeddings and create documents in one go
        return [
            Document(
                text=chunk, 
                vectors={self.model_name: vec} if vec is not None else {}, 
                source=source,
                metadata=extracted_metadata,
                token_count=len(token_chunk) if self.write_token_counts else None,
                fragment_num=i + 1,
                total_fragments=total_fragments
            ) for i, (chunk, token_chunk, vec) in enumerate(zip(
                text_chunks,
                token_chunks,
                self.embed_content(text_chunks, batch_size=self.batch_size, normalize_embeddings=self.normalize_embeddings, **kwargs) if embed else [None] * len(text_chunks)
            ))
        ]
    
    def split_documents(self, documents: List[IDocument], embed: bool = True, **kwargs) -> List[IDocument]:
        ### TODO: fix this
        return [self.split(doc.text, embed=embed, source=doc.source, metadata=doc.metadata) for doc in documents]


    def _content_from_path(self, file_path: Path) -> dict:
        return json.loads(file_path.read_text())

    def extend_text_with_metadata(self, content: str, metadata: dict, source: str | None = None, fragment_num: int = None, total_fragments: int = None) -> str:
        """
        Extends content text with metadata header
        
        Args:
            content: The actual content text
            metadata: Dictionary of metadata fields
            source: Optional source information
            fragment_num: Current fragment number
            total_fragments: Total number of fragments
            
        Returns:
            Text with metadata prepended as markdown header
        """
        header = self.compute_metadata_header(metadata, source, fragment_num, total_fragments)
        if header:
            return header + content
        return content

class DictionarySplitter(SentenceTransformerMixin, AbstractDictionarySplitter):
    pass

class RemoteDictionarySplitter(SentenceTransformerMixin, AbstractDictionarySplitter):
    pass