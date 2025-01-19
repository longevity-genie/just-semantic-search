from just_semantic_search.text_splitters import DEFAULT_MINIMAL_TOKENS, DEFAULT_SIMILARITY_THRESHOLD, AbstractSplitter
from typing import List, TypeAlias, Generic
from pathlib import Path
from just_semantic_search.document import ArticleDocument, Document, IDocument
from pydantic import Field
from sentence_transformers import util


class ParagraphTextSplitter(AbstractSplitter[List[str], IDocument], Generic[IDocument]):
    """Implementation of AbstractSplitter for lists of paragraphs that works with any Document type."""
    

    def should_add_paragraph(
        self,
        current_text: str,
        current_token_count: int,
        new_paragraph: str, #will be used in overides
        new_token_count: int,
        metadata_overhead: int,
        max_tokens: int
    ) -> bool:
        # First check if this is the first paragraph
        if current_text == "":
            return True
        
        # Check if adding would exceed token limit
        return (current_token_count + new_token_count + metadata_overhead) <= max_tokens
    


    def split(self, content: List[str], embed: bool = True, source: str | None = None, **kwargs) -> List[IDocument]:
        # Use batch tokenization:

        metadata_overhead = self.document_type.metadata_overhead(self.tokenizer, **kwargs)
        token_counts = [len(self.tokenizer.tokenize(p)) for p in content]
        chunks: list[str] = []
        chunk_token_counts: list[int] = []
        current_text: str = ""
        current_token_count: int = 0 # current token count of current chunk
        max_tokens = self.max_seq_length
        
        for i, (paragraph, token_count) in enumerate(zip(content, token_counts)):
            should_add = self.should_add_paragraph(
                current_text=current_text,
                current_token_count=current_token_count,
                new_paragraph=paragraph,
                new_token_count=token_count,
                metadata_overhead=metadata_overhead,
                max_tokens=max_tokens
            )
            
            if should_add:
                current_text += f"\n\n{paragraph}"
                current_token_count += token_count
            else:
                if current_text!="":
                    chunks.append(current_text)
                    chunk_token_counts.append(current_token_count)
                current_text = paragraph
                current_token_count = token_count

        # Add final chunk if any remains
        if current_text!="":
            chunks.append(current_text)
            chunk_token_counts.append(current_token_count)

        # Generate embeddings one chunk at a time if requested
        vectors = [self.embed_content([chunk], batch_size=1, normalize_embeddings=self.normalize_embeddings, **kwargs)[0] 
                  for chunk in chunks] if embed else [None] * len(chunks)
        
        # Create documents
        results = [self.document_type.model_validate({
            'text': text,
            'vectors': {self.model_name: vec.tolist()} if vec is not None else {},
            'source': source,
            'token_count': count if self.write_token_counts else None,
            'fragment_num': i + 1,
            'total_fragments': len(chunks),
            **kwargs
        }) for i, (text, vec, count) in enumerate(zip(chunks, vectors, chunk_token_counts))]
        total_tokens = sum(token_counts)
        documents_total_tokens = sum([d.token_count for d in results])
        assert documents_total_tokens >= total_tokens and documents_total_tokens <= total_tokens + metadata_overhead * len(results), f"Total tokens: {documents_total_tokens} must be greater than or equal to {sum(token_counts)} and less than or equal to {sum(token_counts) + metadata_overhead}"
        return results

    def _content_from_path(self, file_path: Path) -> List[str]:
        """Load content from file as list of paragraphs."""
        text = file_path.read_text(encoding="utf-8")
        # Split on double newlines to get paragraphs
        return [p.strip() for p in text.split('\n\n') if p.strip()]

# Type alias for convenience
DocumentParagraphSplitter: TypeAlias = ParagraphTextSplitter[Document]

class ArticleParagraphSplitter(ParagraphTextSplitter[ArticleDocument]):
    """
    A specialized paragraph splitter for articles that uses semantic similarity
    to determine paragraph grouping while respecting token limits and metadata.
    """
        
    @property
    def document_type(self) -> type[ArticleDocument]:
        return ArticleDocument


class ParagraphSemanticSplitter(ParagraphTextSplitter[IDocument], Generic[IDocument]):
    similarity_threshold: float = Field(default=DEFAULT_SIMILARITY_THRESHOLD)
    min_token_count: int = Field(default=DEFAULT_MINIMAL_TOKENS)

    def should_add_paragraph(
            self,
            current_text: str,
            current_token_count: int,
            new_paragraph: str, #will be used in overides
            new_token_count: int,
            metadata_overhead: int,
            max_tokens: int
        ) -> bool:
            # First check if this is the first paragraph
            if current_text == "":
                return True

            summed_tokens = current_token_count + new_token_count + metadata_overhead

            # Check if adding would exceed token limit
            if summed_tokens > max_tokens:
                return False

            if summed_tokens < self.min_token_count:
                return True

            # Check semantic similarity with the last paragraph
            # TODO: decide if we need to use the last paragraph or all of them
            similarity = self.similarity(current_text, new_paragraph)
            return similarity >= self.similarity_threshold

    def similarity(self, text1: str, text2: str, **kwargs) -> float:
        kwargs.update(self.model_params.separatation)
        try:
            vec1 = self.model.encode(text1, batch_size=self.batch_size, normalize_embeddings=self.normalize_embeddings, convert_to_numpy=True, **kwargs).reshape(1, -1)
            vec2 = self.model.encode(text2, batch_size=self.batch_size, normalize_embeddings=self.normalize_embeddings, convert_to_numpy=True, **kwargs).reshape(1, -1)
            return util.cos_sim(vec1, vec2)[0][0]
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
        


ParagraphSemanticDocumentSplitter: TypeAlias = ParagraphSemanticSplitter[Document]

class ArticleSemanticParagraphSplitter(ParagraphSemanticSplitter[ArticleDocument]):
    """
    A specialized paragraph splitter for articles that uses semantic similarity
    to determine paragraph grouping while respecting token limits and metadata.
    """

   
        
    @property
    def document_type(self) -> type[ArticleDocument]:
        return ArticleDocument