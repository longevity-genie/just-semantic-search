from sentence_transformers import SentenceTransformer
from typing import List, Tuple, TypeAlias, TypeVar, Generic, Optional, Any, Type
import numpy as np
from pathlib import Path
import re
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer
from just_semantic_search.document import Document, IDocument
from multiprocessing import Pool, cpu_count
import torch
import eliot
import time
from datetime import datetime
from eliot import log_call, log_message, start_action, Message, Action, preserve_context, ActionType, Field
import logging
from just_semantic_search.utils.logs import LogLevel
from just_semantic_search.utils.models import get_sentence_transformer_model_name


# Define type variables for input and output types
CONTENT = TypeVar('CONTENT')  # Generic content type




class AbstractSplitter(ABC, Generic[CONTENT, IDocument]):
    """Abstract base class for splitting content into documents with optional embedding."""

    
    def __init__(self, 
                 model: SentenceTransformer, 
                 max_seq_length: int | None = None, 
                 tokenizer: Optional[PreTrainedTokenizer | Any] = None, 
                 model_name: Optional[str] = None, 
                 write_token_counts: bool = True,
                 batch_size: int = 32,
                 normalize_embeddings: bool = False):
        """
        Initialize splitter with a transformer model and optional parameters.
        Args:
            model: SentenceTransformer model for text processing
            max_seq_length: Maximum sequence length for tokenization
            tokenizer: Custom tokenizer (uses model's tokenizer if None)
        """
        self.write_token_counts = write_token_counts
        self.model = model
        self.model_name = get_sentence_transformer_model_name(model) if model_name is None else model_name
    
        if tokenizer is None:
            tokenizer = self.model.tokenizer
        self.tokenizer = tokenizer
        if max_seq_length is None:
            self.max_seq_length = self.model.max_seq_length
        else:
            self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

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
            
        with start_action(action_type="processing_file", file_path=str(file_path.absolute()), log_level=LogLevel.DEBUG) as action:
            content: CONTENT = self._content_from_path(file_path)
            documents = self.split(content, embed, 
                               source=str(file_path.absolute()) if path_as_source else file_path.name,
                               **kwargs)
            action.add_success_fields(num_documents=len(documents))
            return documents

    def split_folder(self, folder_path: Path | str, embed: bool = True, path_as_source: bool = True, **kwargs) -> List[IDocument]:
        """Split all files in a folder into documents."""
        with start_action(action_type="split_folder", folder_path=str(folder_path.absolute()), log_level=LogLevel.DEBUG, embed=embed, path_as_source=path_as_source) as action:
            start_time = time.time()
            folder_path = Path(folder_path) if isinstance(folder_path, str) else folder_path
        
            # Log the folder path separately as a string
            action.log(message_type="processing_folder", folder_path=str(folder_path.absolute()), log_level=LogLevel.DEBUG)
            
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
                num_documents=len(documents),
                log_level=LogLevel.INFO
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
        log_message(message_type="processing_batched_folder", folder_path=str(folder_path.absolute()), log_level=LogLevel.DEBUG)
        
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
            total_documents=sum(len(batch) for batch in batches),
            log_level=LogLevel.INFO
        )
            
        return batches



class TextSplitter(AbstractSplitter[str, IDocument], Generic[IDocument]):
    """Implementation of AbstractSplitter for text content that works with any Document type."""
    
    def split(self, text: str, embed: bool = True, source: str | None = None, **kwargs) -> List[IDocument]:
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
        vectors = self.model.encode(text_chunks, batch_size=self.batch_size, normalize_embeddings=self.normalize_embeddings) if embed else [None] * len(text_chunks)
        
        # Create documents using Pydantic's model_validate
        return [IDocument.model_validate({
            'text': text,
            'vectors': {self.model_name: vec.tolist()} if vec is not None else {},
            'source': source,
            'token_count': len(tokens) if self.write_token_counts else None,
            **kwargs
        }) for text, vec in zip(text_chunks, vectors)]
    

    def _content_from_path(self, file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8")
    
    def _encode(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)
    

# Option 1: Type alias
DocumentTextSplitter: TypeAlias = TextSplitter[Document]
