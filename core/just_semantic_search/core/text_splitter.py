from sentence_transformers import SentenceTransformer
from typing import List, Tuple, TypeVar, Generic, Optional, Any, Type
import numpy as np
from pathlib import Path
import re
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer
from just_semantic_search.core.document import Document


# Define type variables for input and output types
CONTENT = TypeVar('CONTENT')  # Generic content type
IDocument = TypeVar('IDocument', bound=Document)  # Document type that must inherit from Document class

class AbstractSplitter(ABC, Generic[CONTENT, IDocument]):
    """Abstract base class for splitting content into documents with optional embedding."""
    
    def __init__(self, model: SentenceTransformer, max_seq_length: int | None = None, tokenizer: Optional[PreTrainedTokenizer | Any] = None):
        """
        Initialize splitter with a transformer model and optional parameters.
        Args:
            model: SentenceTransformer model for text processing
            max_seq_length: Maximum sequence length for tokenization
            tokenizer: Custom tokenizer (uses model's tokenizer if None)
        """
        self.model = model
        if tokenizer is None:
            tokenizer = self.model.tokenizer
        self.tokenizer = tokenizer
        if max_seq_length is None:
            self.max_seq_length = self.model.max_seq_length
        else:
            self.max_seq_length = max_seq_length

    @abstractmethod
    def split(self, content: CONTENT, embed: bool = True, **kwargs) -> List[IDocument]:
        """Split content into documents and optionally embed them."""
        pass

    @abstractmethod
    def _content_from_path(self, file_path: Path) -> CONTENT:
        """Load content from a file path."""
        pass

    def split_file(self, file_path: Path | str, embed: bool = True, **kwargs) -> List[IDocument]:
        """
        Convenience method to split content directly from a file.
        Converts string paths to Path objects and delegates to split().
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        content: CONTENT = self._content_from_path(file_path)
        return self.split(content, embed, **kwargs)


class TextSplitter(AbstractSplitter[str, Document]):
    """Implementation of AbstractSplitter for text content."""
    
    def split(self, text: str, embed: bool = True, **kwargs) -> List[Document]:
        """
        Split text into chunks based on token length.
        Note: Current implementation has an undefined max_seq_length variable
        and doesn't create Document objects as specified in return type.
        """
        # Get the tokenizer from the model
        tokenizer = self.model.tokenizer

        # Tokenize the entire text
        tokens = tokenizer.tokenize(text)

        # Split tokens into chunks of max_seq_length
        token_chunks = [tokens[i:i + self.max_seq_length] for i in range(0, len(tokens), self.max_seq_length)]
        
        # Convert token chunks back to text
        text_chunks = [tokenizer.convert_tokens_to_string(chunk) for chunk in token_chunks]
        
        # Generate embeddings if requested
        vectors = self.model.encode(text_chunks) if embed else [None] * len(text_chunks)
        
        # Create Document objects directly from the two lists
        return [Document(content=text, vectors=vec) for text, vec in zip(text_chunks, vectors)]
    
    def _content_from_path(self, file_path: Path) -> str:
        """Read text content from a file with UTF-8 encoding."""
        return file_path.read_text(encoding="utf-8")
    
