from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from just_semantic_search.document import ArticleDocument
import torch
from transformers import PreTrainedTokenizer
from just_semantic_search.text_splitters import *
import re



class ArticleSemanticSplitter(SemanticSplitter[ArticleDocument]):
    """A semantic splitter specialized for articles that maintains document structure and metadata."""
    
    def split(
        self, 
        content: str, 
        embed: bool = True, 
        source: str = None,
        title: str = None,
        abstract: str = None,
        **kwargs
    ) -> List[ArticleDocument]:
        # Get parameters and calculate adjusted chunk size as before
        max_seq_length = kwargs.get('max_seq_length', self.max_seq_length)
        similarity_threshold = kwargs.get('similarity_threshold', self.similarity_threshold)
        
        adjusted_max_chunk_size = ArticleDocument.calculate_adjusted_chunk_size(
            self.tokenizer,
            max_seq_length,
            title=title,
            abstract=abstract,
            source=source
        )
        
        # Split into sections more efficiently
        sections = self._split_into_sections(content)
        
        # Pre-allocate lists
        documents = []
        
        # Process all sections at once
        all_chunks = []
        for section_title, section_content in sections:
            if section_content.strip():
                chunks: list[str] = self.split_text_semantically(
                    section_content,
                    max_chunk_size=adjusted_max_chunk_size,
                    similarity_threshold=similarity_threshold
                )
                all_chunks.extend((section_title, chunk) for chunk in chunks)
        
        # Create all documents at once and calculate token counts
        documents = []
        for i, (section_title, chunk) in enumerate(all_chunks):
            doc = ArticleDocument(
                text=chunk,
                title=title,
                section_title=section_title,
                abstract=abstract,
                source=source,
                fragment_num=i + 1,
                total_fragments=len(all_chunks)
            )
            # Add token count if enabled
            if self.write_token_counts:
                doc.token_count = len(self.tokenizer.tokenize(doc.content))
            documents.append(doc)
        
        if embed:
            vectors = [self.model.encode(doc.content, batch_size=self.batch_size, normalize_embeddings=self.normalize_embeddings, **kwargs) 
                       for doc in documents]
            documents = [doc.with_vector(self.model_name, vec) for doc, vec in zip(documents, vectors)]

        
        return documents
    
    def _split_into_sections(self, content: str) -> List[Tuple[str, str]]:
        # More efficient header pattern matching
        header_pattern = re.compile(r'^(?:#{1,6}\s+)?([A-Z][^.\n]{0,98})\n', re.MULTILINE)
        
        # Split content at headers
        sections = []
        last_end = 0
        current_title = None
        
        matches = list(header_pattern.finditer(content))
        
        # If no headers found, treat entire content as one section
        if not matches:
            return [("Main Text", content.strip())]
        
        # Process sections with headers
        for match in matches:
            if last_end > 0:  # Not the first section
                sections.append((current_title, content[last_end:match.start()].strip()))
            current_title = match.group(1).strip()
            last_end = match.end()
        
        # Add the final section
        if last_end < len(content):
            sections.append((current_title, content[last_end:].strip()))
        
        return sections
    

    def split_file(self, file_path: Path | str, embed: bool = True, 
                   title: str | None = None,
                   abstract: str | None = None,
                   source: str | None = None,  
                   **kwargs) -> List[ArticleDocument]:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if source is None:
            source = str(file_path.absolute())
        content: str = self._content_from_path(file_path)
        return self.split(content, embed, title=title, abstract=abstract, source=source, **kwargs)
    
    @property
    def document_type(self) -> type[ArticleDocument]:
        return ArticleDocument
    