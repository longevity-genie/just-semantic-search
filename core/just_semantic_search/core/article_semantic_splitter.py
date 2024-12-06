from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from just_semantic_search.core.document import ArticleDocument
from transformers import PreTrainedTokenizer
from just_semantic_search.core.semanic_splitter import *




class ArticleSemanticSplitter(SemanticSplitter):

    def __init__(
        self, 
        model: SentenceTransformer,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        max_seq_length: Optional[int] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        super().__init__(model, similarity_threshold, max_seq_length, tokenizer)

    def split(
        self, 
        content: str, 
        embed: bool = True, 
        source: str = None,
        title: str = None,
        abstract: str = None,
        **kwargs
    ) -> List[ArticleDocument]:
        # Get parameters from kwargs or use defaults
        max_seq_length = kwargs.get('max_seq_length', self.max_seq_length)
        similarity_threshold = kwargs.get('similarity_threshold', self.similarity_threshold)
        
        # Calculate adjusted chunk size
        adjusted_max_chunk_size = ArticleDocument.calculate_adjusted_chunk_size(
            self.tokenizer,
            max_seq_length,
            title=title,
            abstract=abstract,
            source=source
        )
        
        # Split the text into chunks using semantic splitting
        text_chunks = self.split_text_semantically(
            content,
            max_chunk_size=adjusted_max_chunk_size,
            similarity_threshold=similarity_threshold
        )
        
        # Generate embeddings if requested
        vectors = self.model.encode(text_chunks) if embed else [None] * len(text_chunks)
        
        # Create annotated ArticleDocument objects
        documents = []
        for i, (chunk, vector) in enumerate(zip(text_chunks, vectors)):
            doc = ArticleDocument(
                content=chunk,
                vectors=vector,
                title=title,
                abstract=abstract,
                source=source,
                fragment_num=i + 1,
                total_fragments=len(text_chunks)
            )
            documents.append(doc.with_extended_content())
        
        return documents
    
    def _content_from_path(self, file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8")

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