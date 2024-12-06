from sentence_transformers import SentenceTransformer
from typing import List, TypeVar
from just_semantic_search.core.text_splitter import AbstractSplitter
from pathlib import Path
from just_semantic_search.core.document import Document, ArticleDocument
# Add at the top of the file, after imports
DEFAULT_SIMILARITY_THRESHOLD = 0.92



class ArticleSplitter(AbstractSplitter[str, ArticleDocument]):
    """
    A specialized text splitter designed for processing scientific articles and research papers.
    
    This splitter creates ArticleDocument objects that maintain the document's structure with
    title, abstract, and source information. It's particularly useful for:
    - Processing academic papers and research articles
    - Maintaining document metadata (title, abstract) during splitting
    - Creating embeddings for scientific content while preserving context
    
    The splitter ensures that the resulting chunks are properly sized for the underlying
    transformer model while maintaining document attribution.
    """

    def __init__(self, model: SentenceTransformer):
        super().__init__(model)
    

    def split(self, text: str, embed: bool = True, 
              title: str | None = None,
              abstract: str | None = None,
              source: str | None = None,  
              **kwargs) -> List[Document]:
        """
        Split text into chunks based on token length.
        Note: Current implementation has an undefined max_seq_length variable
        and doesn't create Document objects as specified in return type.
        """
        adjusted_max_chunk_size = Document.calculate_adjusted_chunk_size(
                self.model.tokenizer,
                self.max_seq_length,
                title=title,
                abstract=abstract,
                source=source
            )

        # Get the tokenizer from the model
        tokenizer = self.model.tokenizer

        # Tokenize the entire text
        tokens = tokenizer.tokenize(text)

        # Split tokens into chunks of max_seq_length
        token_chunks = [tokens[i:i + self.max_seq_length] for i in range(0, len(tokens), adjusted_max_chunk_size)]
        
        # Convert token chunks back to text
        text_chunks = [tokenizer.convert_tokens_to_string(chunk) for chunk in token_chunks]
        
        # Generate embeddings if requested
        vectors = self.model.encode(text_chunks) if embed else [None] * len(text_chunks)
        
        # Create Document objects directly from the two lists
        return [ArticleDocument(content=text, vectors=vec, title=title, abstract=abstract, source=source).with_extended_content() for text, vec in zip(text_chunks, vectors)]

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