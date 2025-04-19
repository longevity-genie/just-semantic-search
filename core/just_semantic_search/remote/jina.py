import os
from dotenv import load_dotenv
import requests
from pydantic import BaseModel, field_validator
from typing import List, Literal, Optional, Any, Dict, Tuple
from enum import Enum


class JinaTask(str, Enum):
    QUERY = "retrieval.query"
    PASSAGE = "retrieval.passage"
    SEPARATION = "separation"
    CLASSIFICATION = "classification"
    TEXT_MATCHING = "text-matching"


class JinaUsage(BaseModel):
    total_tokens: Optional[int] = None  # Made optional to handle different API responses
    prompt_tokens: Optional[int] = None # rerank usage doesn't have prompt_tokens
    tokens: Optional[int] = None # tokenize usage has tokens field


class JinaEmbeddingData(BaseModel):
    object: Literal["embedding"]
    index: int
    embedding: List[float]


class JinaEmbeddingResponse(BaseModel):
    model: str
    object: Literal["list"]
    usage: JinaUsage
    data: List[JinaEmbeddingData]

    def first_embedding(self) -> List[float]:
        return self.data[0].embedding


# Removed JinaDocument Model

# Removed RerankResult and JinaRerankResponse classes

# Models for tokenize endpoint
class JinaTokenData(BaseModel):
    """Represents a token and its encoding IDs"""
    token: str
    ids: List[int]


class JinaTokenizeResponse(BaseModel):
    """Response model for the Jina AI segment API"""
    num_tokens: int
    tokenizer: str
    usage: JinaUsage
    num_chunks: Optional[int] = None
    chunk_positions: Optional[List[List[int]]] = None
    tokens: Optional[Any] = None  # Changed to Any to handle different response structures
    chunks: Optional[List[str]] = None


def jina_embed_raw(text: str | list[str], model: str = "jina-embeddings-v3", task: str = "retrieval.query") -> JinaEmbeddingResponse:

    load_dotenv()
    key = os.getenv("JINA_API_KEY")
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {key}',
    }
    input = text if isinstance(text, list) else [text]

    data = {
        "model": model,
        "task": task,
        "input": input
    }

    response = requests.post('https://api.jina.ai/v1/embeddings', headers=headers, json=data)
    response.raise_for_status()
    return JinaEmbeddingResponse.model_validate(response.json())

def jina_embed_query(text: str | list[str], model: str = "jina-embeddings-v3") -> List[float]:
    response = jina_embed_raw(text, model, "retrieval.query")
    return response.first_embedding()

def jina_embed_passage(text: str | list[str], model: str = "jina-embeddings-v3") -> List[float]:
    response = jina_embed_raw(text, model, "retrieval.passage")
    return response.first_embedding()


# Removed jina_rerank_raw and jina_rerank functions


def jina_tokenize(content: str, return_tokens: bool = True, return_chunks: bool = True, 
                 max_chunk_length: Optional[int] = None) -> JinaTokenizeResponse:
    """
    Tokenize and chunk text using Jina AI's segment API.
    
    Args:
        content: The text to tokenize and chunk
        return_tokens: Whether to return token information
        return_chunks: Whether to return text chunks
        max_chunk_length: Maximum length of each chunk in tokens
        
    Returns:
        A JinaTokenizeResponse object containing tokenization and chunking results
    """
    load_dotenv()
    key = os.getenv("JINA_API_KEY")
    url = 'https://api.jina.ai/v1/segment'
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {key}'
    }
    
    data = {
        'content': content,
        'return_tokens': return_tokens,
        'return_chunks': return_chunks
    }
    
    if max_chunk_length is not None:
        data['max_chunk_length'] = max_chunk_length
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    return JinaTokenizeResponse.model_validate(response.json())

if __name__ == "__main__":
    response = jina_tokenize("Hello, world!", return_tokens=True, return_chunks=True)
    print(response)