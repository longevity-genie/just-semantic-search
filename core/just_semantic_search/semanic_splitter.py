from just_semantic_search.document import Document, IDocument
from just_semantic_search.text_splitter import AbstractSplitter, TextSplitter
from sentence_transformers import SentenceTransformer
from typing import Generic, List, Optional, TypeAlias
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import PreTrainedTokenizer
from pathlib import Path

# Add at the top of the file, after imports
DEFAULT_SIMILARITY_THRESHOLD = 0.60
DEFAULT_MINIMAL_TOKENS = 500


class SemanticSplitter(TextSplitter[IDocument], Generic[IDocument]):

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

    
    def __init__(
        self, 
        model: SentenceTransformer, 
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        max_seq_length: Optional[int] = None,
        min_token_count: int = DEFAULT_MINIMAL_TOKENS,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model_name: Optional[str] = None,
        write_token_counts: bool = True,
        batch_size: int = 32,
        normalize_embeddings: bool = False
    ):
        # First call parent's init with correct parameter order
        super().__init__(model, max_seq_length, tokenizer, model_name, write_token_counts, batch_size, normalize_embeddings)
        # Then set additional parameters specific to SemanticSplitter
        self.similarity_threshold = similarity_threshold
        self.min_token_count = min_token_count

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
        
        # Generate embeddings if requested
        vectors = self.model.encode(text_chunks) if embed else [None] * len(text_chunks)
        
        # Create Document objects
        return [Document(text=text, vectors={ self.model_name: vec }, source=source) for text, vec in zip(text_chunks, vectors)]


    def similarity(self, text1: str, text2: str) -> float:
        try:
            vec1 = self.model.encode(text1, convert_to_numpy=True).reshape(1, -1)
            vec2 = self.model.encode(text2, convert_to_numpy=True).reshape(1, -1)
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


    def similarity_batch(self, texts: List[str]) -> np.ndarray:
        """Calculate similarity matrix for a batch of texts"""
        # Encode all texts at once
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        # Calculate similarity matrix
        return cosine_similarity(embeddings)
    

SemanticDocumentSplitter: TypeAlias = SemanticSplitter[Document]