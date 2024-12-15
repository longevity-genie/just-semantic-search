from just_semantic_search.document import Document
from just_semantic_search.text_splitter import AbstractSplitter
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import PreTrainedTokenizer
from pathlib import Path

# Add at the top of the file, after imports
DEFAULT_SIMILARITY_THRESHOLD = 0.95


class SemanticSplitter(AbstractSplitter[str, Document]):
    
    def __init__(
        self, 
        model: SentenceTransformer, 
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        max_seq_length: Optional[int] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        super().__init__(model, max_seq_length, tokenizer)
        self.similarity_threshold = similarity_threshold

    def split(self, content: str, embed: bool = True, **kwargs) -> List[Document]:
        # Get parameters from kwargs or use defaults
        max_seq_length = kwargs.get('max_seq_length', self.max_seq_length)
        similarity_threshold = kwargs.get('similarity_threshold', self.similarity_threshold)
        
        # Split the text into chunks
        text_chunks = self.split_text_semantically(
            content,
            max_chunk_size=max_seq_length,
            similarity_threshold=similarity_threshold
        )
        
        # Generate embeddings if requested
        vectors = self.model.encode(text_chunks) if embed else [None] * len(text_chunks)
        
        # Create Document objects
        return [Document(content=text, vectors=vec) for text, vec in zip(text_chunks, vectors)]

    def _content_from_path(self, file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8")
    
    def _encode(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)
    
    def similarity(self, text1: str, text2: str) -> float:
        # Encode and reshape vectors to 2D arrays
        vec1 = self.model.encode(text1, convert_to_numpy=True).reshape(1, -1)
        vec2 = self.model.encode(text2, convert_to_numpy=True).reshape(1, -1)
        return cosine_similarity(vec1, vec2)[0][0]

    def split_text_semantically(
        self,
        text: str,
        max_chunk_size: int | None = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[str]:
        # Use model's max sequence length if max_chunk_size is None
        if max_chunk_size is None:
            max_chunk_size = self.model.max_seq_length

        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold

        # First, replace hyphenated line breaks with the complete word
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        
        # Replace simple split with regex-based splitting that preserves gene IDs
        sentence_pattern = r'(?<![A-Za-z0-9])[.!?](?=\s+[A-Z]|$)'
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() + "." for s in sentences if s.strip()]
        
        
        # Initialize chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            # Get token count for current sentence
            sentence_tokens = len(self.model.tokenizer.tokenize(sentence))
            
            if current_length + sentence_tokens > max_chunk_size and current_chunk:
                # Save current chunk and start new one
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # If current chunk is empty, add sentence directly
            if not current_chunk:
                current_chunk.append(sentence)
                current_length += sentence_tokens
                continue
                
            # Check semantic similarity with the last sentence in current chunk
            similarity = self.similarity(sentence, current_chunk[-1])
            
            # If similar enough and within size limit, add to current chunk
            if similarity >= similarity_threshold and current_length + sentence_tokens <= max_chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_tokens
            else:
                # Save current chunk and start new one with current sentence
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_tokens
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks