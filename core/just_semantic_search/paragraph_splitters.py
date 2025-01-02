from just_semantic_search.text_splitter import AbstractSplitter
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, TypeAlias, TypeVar, Generic, Optional, Any, Type
import numpy as np
from pathlib import Path
import re
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer
from just_semantic_search.document import ArticleDocument, Document, IDocument
from multiprocessing import Pool, cpu_count
import torch
import eliot
import time
from datetime import datetime
from eliot import log_call, log_message, start_action, Message, Action, preserve_context, ActionType
import logging
from just_semantic_search.utils.logs import LogLevel
from just_semantic_search.utils.models import get_sentence_transformer_model_name
from pydantic import BaseModel, ConfigDict, Field

class ParagraphTextSplitter(AbstractSplitter[List[str], IDocument], Generic[IDocument]):
    """Implementation of AbstractSplitter for lists of paragraphs that works with any Document type."""
    

    def _should_add_paragraph(
        self,
        current_paragraphs: List[str],
        new_paragraph: str,
        current_token_count: int,
        new_token_count: int,
        max_tokens: int
    ) -> bool:
        # First check if this is the first paragraph
        if not current_paragraphs:
            # If the single paragraph exceeds max tokens, we still need to include it
            # as its own chunk (it will be truncated later)
            return True
        
        # Check if adding would exceed token limit
        return (current_token_count + new_token_count) <= max_tokens

    def split(self, content: List[str], embed: bool = True, source: str | None = None, **kwargs) -> List[IDocument]:
        # Go back to using tokenize() which correctly handles the max token count
        token_counts = [len(self.tokenizer.tokenize(p)) for p in content]
        
        chunks = []
        chunk_token_counts = []
        current_chunk = []
        current_token_count = 0
        
        for i, (paragraph, token_count) in enumerate(zip(content, token_counts)):
            should_add = self._should_add_paragraph(
                current_chunk,
                paragraph,
                current_token_count,
                token_count,
                self.max_seq_length
            )
            
            if should_add:
                current_chunk.append(paragraph)
                current_token_count += token_count
            else:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    chunk_token_counts.append(current_token_count)
                current_chunk = [paragraph]
                current_token_count = token_count

        # Add final chunk if any remains
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
            chunk_token_counts.append(current_token_count)

        # Generate embeddings if requested
        vectors = self.model.encode(chunks, batch_size=self.batch_size, normalize_embeddings=self.normalize_embeddings) if embed else [None] * len(chunks)
        
        # Create documents
        return [self.document_type.model_validate({
            'text': text,
            'vectors': {self.model_name: vec.tolist()} if vec is not None else {},
            'source': source,
            'token_count': count if self.write_token_counts else None,
            'fragment_num': i + 1,
            'total_fragments': len(chunks),
            **kwargs
        }) for i, (text, vec, count) in enumerate(zip(chunks, vectors, chunk_token_counts))]

    def _content_from_path(self, file_path: Path) -> List[str]:
        """Load content from file as list of paragraphs."""
        text = file_path.read_text(encoding="utf-8")
        # Split on double newlines to get paragraphs
        return [p.strip() for p in text.split('\n\n') if p.strip()]

# Type alias for convenience
DocumentParagraphSplitter: TypeAlias = ParagraphTextSplitter[Document]



class ParagraphSemanticSplitter(ParagraphTextSplitter[IDocument], Generic[IDocument]):
    similarity_threshold: float = Field(default=DEFAULT_SIMILARITY_THRESHOLD)
    min_token_count: int = Field(default=DEFAULT_MINIMAL_TOKENS)

    def _should_add_paragraph(
        self,
        current_paragraphs: List[str],
        new_paragraph: str,
        current_token_count: int,
        new_token_count: int,
        max_tokens: int
    ) -> bool:
        
       # First check if this is the first paragraph
        if not current_paragraphs:
            return True

        # Check if adding would exceed adjusted token limit
        if current_token_count + new_token_count > max_tokens:
            return False

        # If below minimum token count, always add
        if current_token_count < self.min_token_count:
            return True
            
        # Check semantic similarity with the last paragraph
        similarity = self.similarity(current_paragraphs[-1], new_paragraph)
        return similarity >= self.similarity_threshold

    def similarity(self, text1: str, text2: str) -> float:
        try:
            vec1 = self.model.encode(text1, convert_to_numpy=True).reshape(1, -1)
            vec2 = self.model.encode(text2, convert_to_numpy=True).reshape(1, -1)
            return cosine_similarity(vec1, vec2)[0][0]
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
        


ParagraphSemanticDocumentSplitter: TypeAlias = ParagraphSemanticSplitter[Document]

class ArticleSemanticParagraphSplitter(ParagraphSemanticSplitter[ArticleDocument]):
    """
    A specialized paragraph splitter for articles that uses semantic similarity
    to determine paragraph grouping while respecting token limits and metadata.
    """


    def _should_add_paragraph(
        self,
        current_paragraphs: List[str],
        new_paragraph: str,
        current_token_count: int,
        new_token_count: int,
        max_tokens: int,
        title: str | None = None,
        abstract: str | None = None,
        source: str | None = None,
        references: str | None = None
    ) -> bool:
        # First check if this is the first paragraph
        if not current_paragraphs:
            return True

        # Calculate adjusted max tokens accounting for metadata
        adjusted_max_tokens = ArticleDocument.calculate_adjusted_chunk_size(
            self.tokenizer,
            max_tokens,
            title=title,
            abstract=abstract,
            source=source,
            references=references
        )

        # Check if adding would exceed token limit
        if (current_token_count + new_token_count) > adjusted_max_tokens:
            return False

        # Check semantic similarity with the last paragraph
        similarity = self.similarity(current_paragraphs[-1], new_paragraph)
        return similarity >= self.similarity_threshold
    
    @property
    def document_type(self) -> type[ArticleDocument]:
        return ArticleDocument