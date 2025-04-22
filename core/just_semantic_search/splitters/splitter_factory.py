from enum import Enum, auto
from typing import Optional, Union
from just_semantic_search.embeddings import EmbeddingModel, load_sentence_transformer_from_enum
from just_semantic_search.splitters.structural_splitters import DictionarySplitter, RemoteDictionarySplitter
from sentence_transformers import SentenceTransformer
from just_semantic_search.splitters.text_splitters import (
    TextSplitter, 
    SemanticSplitter
)

from just_semantic_search.splitters.article_splitter import ArticleSplitter
from just_semantic_search.splitters.article_semantic_splitter import ArticleSemanticSplitter
from just_semantic_search.splitters.paragraph_splitters import (
    ParagraphTextSplitter,
    ParagraphSemanticSplitter,
    ArticleParagraphSplitter,
    ArticleSemanticParagraphSplitter
)

class SplitterType(Enum):
    """Enum for different types of document splitters"""
    TEXT = auto()
    SEMANTIC = auto()
    ARTICLE = auto()
    ARTICLE_SEMANTIC = auto()
    PARAGRAPH = auto()
    PARAGRAPH_SEMANTIC = auto()
    ARTICLE_PARAGRAPH = auto()
    ARTICLE_PARAGRAPH_SEMANTIC = auto()
    FLAT_JSON = auto()
    FLAT_JSON_REMOTE = auto()
    

def create_splitter(
    splitter_type: SplitterType,
    model: SentenceTransformer | EmbeddingModel,
    batch_size: int = 32,
    normalize_embeddings: bool = False,
    similarity_threshold: float = 0.8,
    min_token_count: int = 500,
    max_seq_length: Optional[int] = None
) -> Union[
    TextSplitter,
    SemanticSplitter,
    ArticleSplitter,
    ArticleSemanticSplitter,
    ParagraphTextSplitter,
    ParagraphSemanticSplitter,
    ArticleParagraphSplitter,
    ArticleSemanticParagraphSplitter,
    DictionarySplitter,
    RemoteDictionarySplitter]:
    """
    Factory function to create document splitters based on type.
    
    Args:
        splitter_type: Type of splitter to create from SplitterType enum
        model: SentenceTransformer model to use for embeddings
        batch_size: Batch size for encoding
        normalize_embeddings: Whether to normalize embeddings
        similarity_threshold: Threshold for semantic similarity (for semantic splitters)
        min_token_count: Minimum token count (for semantic splitters)
        
    Returns:
        Configured splitter instance of the requested type
    """
    model: SentenceTransformer = load_sentence_transformer_from_enum(model) if isinstance(model, EmbeddingModel) else model
    
    common_kwargs = {
        "model": model,
        "batch_size": batch_size,
        "normalize_embeddings": normalize_embeddings
    }
    
    if max_seq_length is not None:
        common_kwargs["max_seq_length"] = max_seq_length
    
    semantic_kwargs = {
        **common_kwargs,
        "similarity_threshold": similarity_threshold,
        "min_token_count": min_token_count
    }
    
    splitters = {
        SplitterType.TEXT: lambda: TextSplitter(**common_kwargs),
        SplitterType.SEMANTIC: lambda: SemanticSplitter(**semantic_kwargs),
        SplitterType.ARTICLE: lambda: ArticleSplitter(**common_kwargs),
        SplitterType.ARTICLE_SEMANTIC: lambda: ArticleSemanticSplitter(**semantic_kwargs),
        SplitterType.PARAGRAPH: lambda: ParagraphTextSplitter(**common_kwargs),
        SplitterType.PARAGRAPH_SEMANTIC: lambda: ParagraphSemanticSplitter(**semantic_kwargs),
        SplitterType.ARTICLE_PARAGRAPH: lambda: ArticleParagraphSplitter(**common_kwargs),
        SplitterType.ARTICLE_PARAGRAPH_SEMANTIC: lambda: ArticleSemanticParagraphSplitter(**semantic_kwargs),
        SplitterType.FLAT_JSON: lambda: DictionarySplitter(**common_kwargs),
        SplitterType.FLAT_JSON_REMOTE: lambda: RemoteDictionarySplitter(**common_kwargs)
    }
    
    return splitters[splitter_type]()