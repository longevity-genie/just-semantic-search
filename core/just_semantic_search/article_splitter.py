from typing import List, Optional
from just_semantic_search.text_splitters import TextSplitter
from pathlib import Path
from just_semantic_search.document import Document, ArticleDocument
# Add at the top of the file, after imports

import warnings
import torch
from torch import device as torch_device


class ArticleSplitter(TextSplitter[ArticleDocument]):
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
    device: Optional[torch_device] = None

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        # Determine the device from the model
        self.device = next(self.model.parameters()).device
        
        # Check for available CUDA devices
        if hasattr(torch.cuda, 'device_count'):
            cuda_count = torch.cuda.device_count()
        else:
            cuda_count = 0
        
        # Warn if running on CPU (but not other accelerators)
        accelerators = {'cuda', 'triton', 'mps', 'xpu', 'tpu'}
        if not any(acc in self.device.type.lower() for acc in accelerators):
            warnings.warn(
                f"Model is running on CPU (device is {self.device.type}). "
                f"Found {cuda_count} CUDA device(s). For better performance, consider using an accelerator (GPU/TPU/etc).",
                RuntimeWarning
            )
        
    

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
        adjusted_max_chunk_size = ArticleDocument.metadata_overhead(
                self.model.tokenizer,
                title=title,
                abstract=abstract,
                source=source
            )

        # Get the tokenizer from the model
        tokenizer = self.model.tokenizer

        # Tokenize the entire text
        tokens = tokenizer.tokenize(text)

        # Combine both operations in a single loop
        text_chunks = []
        token_counts = []
        for i in range(0, len(tokens), adjusted_max_chunk_size):
            chunk = tokens[i:i + self.max_seq_length]
            text_chunks.append(tokenizer.convert_tokens_to_string(chunk))
            token_counts.append(len(chunk))
        
        # Create annotated ArticleDocument objects with vectors in one go
        documents = [
            ArticleDocument(
                text=chunk,
                title=title,
                abstract=abstract,
                source=source,
                fragment_num=i + 1,
                total_fragments=len(text_chunks),
                token_count=token_counts[i] if self.write_token_counts else None
            ).with_vector(self.model_name, self.model.encode(chunk) if embed else None)
            for i, chunk in enumerate(text_chunks)
        ]
        
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