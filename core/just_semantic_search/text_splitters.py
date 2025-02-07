from just_semantic_search.embeddings import EmbeddingModelParams
from sentence_transformers import SentenceTransformer
from typing import List, TypeAlias, TypeVar, Generic, Optional, Any
import numpy as np
from pathlib import Path
import re
from abc import ABC, abstractmethod
from transformers import PreTrainedTokenizer
from just_semantic_search.document import ArticleDocument, Document, IDocument
from multiprocessing import Pool, cpu_count
import torch
import time
from eliot import log_call, log_message, start_action
from just_semantic_search.utils.models import get_sentence_transformer_model_name
from pydantic import BaseModel, ConfigDict, Field

from just_semantic_search.document import Document, IDocument
from sentence_transformers import SentenceTransformer
from typing import Generic, List, Optional, TypeAlias
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import PreTrainedTokenizer
from pathlib import Path
# Define type variables for input and output types
CONTENT = TypeVar('CONTENT')  # Generic content type


class AbstractSplitter(ABC, BaseModel, Generic[CONTENT, IDocument]):
    """Abstract base class for splitting content into documents with optional embedding."""
    
    model: SentenceTransformer
    max_seq_length: Optional[int] = None
    tokenizer: Optional[PreTrainedTokenizer | Any] = None
    model_name: Optional[str] = None
    write_token_counts: bool = Field(default=True)
    batch_size: int = Field(default=32)
    normalize_embeddings: bool = Field(default=False)
    #extra_embed_arguments: dict = Field(default_factory=dict)
    model_params: EmbeddingModelParams = Field(default_factory=EmbeddingModelParams)
    
    
    @property
    def document_type(self) -> type[IDocument]:
        return Document

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Needed for SentenceTransformer type

    def model_post_init(self, __context) -> None:
        if self.tokenizer is None:
            self.tokenizer = self.model.tokenizer
        if self.max_seq_length is None:
            self.max_seq_length = self.model.max_seq_length
        if self.model_name is None:
            model_value = get_sentence_transformer_model_name(self.model)
            self.model_name = model_value.split("/")[-1].split("\\")[-1] if "/" in model_value or "\\" in model_value else model_value

    @abstractmethod
    def split(self, content: CONTENT, embed: bool = True, source: str | None = None, **kwargs) -> List[IDocument]:
        """Split content into documents and optionally embed them."""
        pass

    @abstractmethod
    def _content_from_path(self, file_path: Path) -> CONTENT:
        """Load content from a file path."""
        pass

    def split_file(self, file_path: Path | str, embed: bool = True, path_as_source: bool = True, **kwargs) -> List[IDocument]:
        """Convenience method to split content directly from a file."""
        if isinstance(file_path, str):
            file_path = Path(file_path)
            
        with start_action(action_type="processing_file", file_path=str(file_path.absolute())) as action:
            content: CONTENT = self._content_from_path(file_path)
            documents = self.split(content, embed, 
                               source=str(file_path.absolute()) if path_as_source else file_path.name,
                               **kwargs)
            action.add_success_fields(num_documents=len(documents))
            return documents

    def split_folder(self, folder_path: Path | str, embed: bool = True, path_as_source: bool = True, **kwargs) -> List[IDocument]:
        """Split all files in a folder into documents."""
        with start_action(action_type="split_folder", folder_path=str(folder_path.absolute()), embed=embed, path_as_source=path_as_source) as action:
            start_time = time.time()
            folder_path = Path(folder_path) if isinstance(folder_path, str) else folder_path
        
            # Log the folder path separately as a string
            action.log(message_type="processing_folder", folder_path=str(folder_path.absolute()))
            
            if not folder_path.exists() or not folder_path.is_dir():
                raise ValueError(f"Invalid folder path: {folder_path}")

            documents = []
            for file_path in folder_path.iterdir():
                if file_path.is_file():
                    documents.extend(self.split_file(file_path, embed, path_as_source, **kwargs))
            
            elapsed_time = time.time() - start_time
            action.log(
                message_type="folder_processing_complete",
                processing_time_seconds=elapsed_time,
                num_documents=len(documents)
            )
                    
            return documents

    @log_call(
        action_type="split_folder_with_batches", 
        include_args=["batch_size", "embed", "path_as_source", "num_processes"],
        include_result=False
    )
    def split_folder_with_batches(
        self, 
        folder_path: Path | str, 
        batch_size: int = 20,
        embed: bool = True, 
        path_as_source: bool = True,
        num_processes: Optional[int] = None,
        **kwargs
    ) -> List[List[IDocument]]:
        """
        NOTE: SO FAR I DID NOT MANAGED TO GET BENEFITS FROM THIS METHOD. PROBABLY DEFAULT SENTENCE TRANSFORMER BATCH SIZE IS ENOUGH.
        """
        start_time = time.time()
        folder_path = Path(folder_path) if isinstance(folder_path, str) else folder_path
        
        # Log the folder path separately as a string
        log_message(message_type="processing_batched_folder", folder_path=str(folder_path.absolute()))
        
        # Validate inputs
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"The folder_path '{folder_path}' does not exist or is not a directory.")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
            
        # Setup processing
        cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if num_processes is None:
            num_processes = min(cpu_count(), max(1, cuda_devices))
        if num_processes < 1:
            raise ValueError("num_processes must be at least 1.")
            
        # Collect and process files
        file_paths = [f for f in folder_path.iterdir() if f.is_file()]
        if not file_paths:
            return []
            
        # Process files
        if num_processes > 1 and cuda_devices > 0:
            with Pool(num_processes) as pool:
                from functools import partial
                process_file = partial(
                    self.split_file, 
                    embed=embed, 
                    path_as_source=path_as_source, 
                    **kwargs
                )
                all_docs = pool.map(process_file, file_paths)
                all_docs = [doc for file_docs in all_docs for doc in file_docs]
        else:
            all_docs = [
                doc
                for file_path in file_paths
                for doc in self.split_file(file_path, embed, path_as_source, **kwargs)
            ]
        
        # Group into batches
        batches = []
        current_batch = []
        for doc in all_docs:
            current_batch.append(doc)
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
        
        if current_batch:
            batches.append(current_batch)
        
        elapsed_time = time.time() - start_time
        log_message(
            message_type="batched_folder_processing_complete",
            processing_time_seconds=elapsed_time,
            num_batches=len(batches),
            total_documents=sum(len(batch) for batch in batches)
        )
            
        return batches
    
    
    def embed_content(self, content: CONTENT, **kwargs) -> np.ndarray:
        kwargs.update(self.model_params.retrival_passage)
        return self.model.encode(content, convert_to_numpy=True, **kwargs)


class TextSplitter(AbstractSplitter[str, IDocument], Generic[IDocument]):
    """Implementation of AbstractSplitter for text content that works with any Document type."""

    
    
    def split(self, text: str, embed: bool = True, source: str | None = None, **kwargs) -> List[IDocument]:
        
        
        # Get the tokenizer from the model
        tokenizer = self.model.tokenizer

        # Tokenize the entire text
        tokens = tokenizer.tokenize(text)

        # Split tokens into chunks of max_seq_length
        token_chunks = [tokens[i:i + self.max_seq_length] for i in range(0, len(tokens), self.max_seq_length)]
        
        # Convert token chunks back to text
        text_chunks = [tokenizer.convert_tokens_to_string(chunk) for chunk in token_chunks]
        

        # Generate embeddings and create documents in one go
        return [
            Document(
                text=text, 
                vectors={self.model_name: vec} if vec is not None else {}, 
                source=source
            ) for text, vec in zip(
                text_chunks, 
                self.embed_content(text_chunks, batch_size=self.batch_size, normalize_embeddings=self.normalize_embeddings, **kwargs) if embed else [None] * len(text_chunks)
            )
        ]
    

    def _content_from_path(self, file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8")
    
    

# Option 1: Type alias
DocumentTextSplitter: TypeAlias = TextSplitter[Document]


# Add at the top of the file, after imports
DEFAULT_SIMILARITY_THRESHOLD = 0.8
DEFAULT_MINIMAL_TOKENS = 500


class SemanticSplitter(TextSplitter[IDocument], Generic[IDocument]):
    similarity_threshold: float = Field(default=DEFAULT_SIMILARITY_THRESHOLD)
    min_token_count: int = Field(default=DEFAULT_MINIMAL_TOKENS)
    extra_separate_arguments: dict = Field(default_factory=dict)
    

    """
    Text Splitting Logic in SemanticSplitter

    The SemanticSplitter class implements a sophisticated text chunking strategy that combines
    semantic similarity with size constraints. Here's how it works:

    1. Primary Split (split_text_semantically):
    - First normalizes the text by:
        * Reducing multiple newlines to double newlines
        * Converting table-like spacing to pipe separators
        * Fixing hyphenated words across lines
    - Splits text into paragraphs using double newlines
    - Processes paragraphs in batches of 5 for efficient similarity computation
    - For single large paragraphs, delegates to sentence-level splitting
    - Otherwise processes paragraphs sequentially, combining them based on:
        * Semantic similarity (must be >= similarity_threshold)
        * Size constraints (must not exceed max_chunk_size in tokens)
        * Minimum token count (won't split if below min_token_count)

    2. Secondary Split (_split_large_text):
    - Used when paragraphs are too large
    - Splits text into sentences using regex pattern
    - Falls back to token-based splitting if sentence splitting fails
    - Combines sentences based on:
        * Semantic similarity
        * Token count constraints

    Key Parameters:
    - similarity_threshold: Minimum cosine similarity (default: 0.60) required to combine chunks
    - max_chunk_size: Maximum number of tokens allowed in a single chunk
    - min_token_count: Minimum tokens required before splitting (default: 500)
    - model: SentenceTransformer model used for encoding text and calculating similarity

    The process ensures that:
    1. Output chunks don't exceed the model's maximum sequence length
    2. Related content stays together based on semantic similarity
    3. Natural text boundaries (paragraphs, sentences) are preserved where possible
    4. Edge cases (very long texts, malformed input) are handled gracefully
    5. Performance is optimized through batch processing
    6. Chunks maintain a minimum size for meaningful analysis
    """

    def split(self, content: str, embed: bool = True, source: str | None = None, **kwargs) -> List[Document]:
        # Get parameters from kwargs or use defaults
        max_seq_length = kwargs.get('max_seq_length', self.max_seq_length)
        similarity_threshold = kwargs.get('similarity_threshold', self.similarity_threshold)
        
        # Split the text into chunks
        text_chunks = self.split_text_semantically(
            content,
            max_chunk_size=max_seq_length,
            similarity_threshold=similarity_threshold
        )
        
        # Generate embeddings and create documents in one go
        return [
            Document(
                text=text, 
                vectors={self.model_name: vec} if vec is not None else {}, 
                source=source
            ) for text, vec in zip(
                text_chunks, 
                self.embed_content(text_chunks, batch_size=self.batch_size, normalize_embeddings=self.normalize_embeddings, **kwargs) if embed else [None] * len(text_chunks)
            )
        ]


    def similarity(self, text1: str, text2: str, **kwargs) -> float:
        try:
            vec1 = self.model.encode(text1, convert_to_numpy=True, batch_size=self.batch_size, normalize_embeddings=self.normalize_embeddings, **kwargs).reshape(1, -1)
            vec2 = self.model.encode(text2, convert_to_numpy=True, batch_size=self.batch_size, normalize_embeddings=self.normalize_embeddings, **kwargs).reshape(1, -1)
            return cosine_similarity(vec1, vec2)[0][0]
        except Exception as e:
            # Log error and return minimum similarity to force split
            print(f"Error calculating similarity: {e}")
            return 0.0

    def split_text_semantically(
        self,
        text: str,
        max_chunk_size: int | None = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[str]:
        """
        Splits text into semantically coherent chunks, handling edge cases like
        multiple empty lines and malformed tables.
        """
        # Input validation
        if not text or not text.strip():
            return []

        if max_chunk_size is None:
            max_chunk_size = self.model.max_seq_length

        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold

        # Check total text length
        total_tokens = len(self.tokenizer.tokenize(text))
        if total_tokens <= self.min_token_count:
            return [text]  # Return whole text as single chunk if it's smaller than min_token_count

        # Normalize whitespace and handle potential table formatting
        text = re.sub(r'\n{3,}', '\n\n', text)  # Replace multiple newlines
        text = re.sub(r'[\t ]{3,}', ' | ', text)  # Handle table-like formatting
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)  # Fix hyphenated words

        # First split by paragraphs (double newlines)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Process paragraphs in batches
        batch_size = 5
        chunks = []
        current_batch = []
        current_length = 0
        
        for i in range(0, len(paragraphs), batch_size):
            batch = paragraphs[i:i + batch_size]
            if len(batch) > 1:
                sim_matrix = self.similarity_batch(batch)
            
            for j, para in enumerate(batch):
                para_tokens = len(self.tokenizer.tokenize(para))
                
                # Check if adding this paragraph would exceed max_chunk_size
                if current_batch and current_length + para_tokens > max_chunk_size:
                    # Only append if we meet minimum token count
                    if current_length >= self.min_token_count:
                        chunks.append("\n\n".join(current_batch))
                        current_batch = [para]
                        current_length = para_tokens
                    else:
                        # If below minimum, keep adding despite similarity
                        current_batch.append(para)
                        current_length += para_tokens
                    continue
                
                # Use pre-computed similarity
                if len(batch) > 1:
                    similarity = sim_matrix[j-1][j] if j > 0 else 0
                else:
                    similarity = 1.0
                    
                if similarity >= similarity_threshold:
                    current_batch.append(para)
                    current_length += para_tokens
                else:
                    # Only create new chunk if we meet minimum token count
                    if current_length >= self.min_token_count:
                        chunks.append("\n\n".join(current_batch))
                        current_batch = [para]
                        current_length = para_tokens
                    else:
                        # If below minimum, keep adding despite similarity
                        current_batch.append(para)
                        current_length += para_tokens
        
        # Handle the last batch
        if current_batch:
            chunks.append("\n\n".join(current_batch))
        
        return chunks

    def _split_large_text(self, text: str, max_chunk_size: int, similarity_threshold: float) -> List[str]:
        """
        Helper method to split large text chunks, first attempting sentence-level splitting,
        then falling back to token-based splitting if needed.
        
        Args:
            text: The text to split
            max_chunk_size: Maximum number of tokens per chunk
            similarity_threshold: Minimum similarity score to combine chunks
            
        Returns:
            List of text chunks that respect token limits and maintain semantic coherence
        """
        # First try sentence splitting for more natural boundaries
        sentence_pattern = r'(?<![A-Za-z0-9])[.!?](?=\s+[A-Z]|$)'
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() + "." for s in sentences if s.strip()]

        # If no sentence boundaries found, fall back to token-based splitting
        # This ensures we always get valid chunks that respect the model's token limits
        if not sentences:
            tokens = self.model.tokenizer.tokenize(text)
            current_chunk = []
            chunks = []
            current_length = 0
            
            for token in tokens:
                if current_length + 1 > max_chunk_size and current_chunk:
                    # Convert accumulated tokens back to coherent text
                    chunk_text = self.model.tokenizer.convert_tokens_to_string(current_chunk)
                    chunks.append(chunk_text)
                    current_chunk = []
                    current_length = 0
                
                current_chunk.append(token)
                current_length += 1
            
            if current_chunk:
                chunk_text = self.model.tokenizer.convert_tokens_to_string(current_chunk)
                chunks.append(chunk_text)
            
            return chunks

        # Process sentences normally, combining them based on semantic similarity
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.model.tokenizer.tokenize(sentence))
            
            if current_length + sentence_tokens > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            if not current_chunk:
                current_chunk.append(sentence)
                current_length += sentence_tokens
                continue
            
            similarity = self.similarity(sentence, current_chunk[-1])
            
            if similarity >= similarity_threshold and current_length + sentence_tokens <= max_chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_tokens
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks


    def similarity_batch(self, texts: List[str], **kwargs) -> np.ndarray:
        """Calculate similarity matrix for a batch of texts"""
        # Encode all texts at once
        kwargs.update(self.model_params.separatation)
        embeddings = self.model.encode(texts, convert_to_numpy=True, **kwargs)
        # Calculate similarity matrix
        return cosine_similarity(embeddings)
    

SemanticDocumentSplitter: TypeAlias = SemanticSplitter[Document]



class ArticleSemanticSplitter(SemanticSplitter[ArticleDocument]):


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

        metadata_overhead = ArticleDocument.metadata_overhead(
            self.tokenizer,
            title=title,
            abstract=abstract,
            source=source
        )
        
        adjusted_max_chunk_size = max_seq_length - metadata_overhead
        
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
        
        # Batch encode all documents at once
        if embed:
            vectors = [self.model.encode(doc.content, batch_size=self.batch_size, normalize_embeddings=self.normalize_embeddings, **kwargs) for doc in documents]
            documents = [doc.with_vector(self.model_name, vec) for doc, vec in zip(documents, vectors)]

        
        return documents
    
    def _split_into_sections(self, content: str) -> List[tuple[str, str]]:
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
    