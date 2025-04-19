import os
from dotenv import load_dotenv
import requests
from pydantic import BaseModel, field_validator
from typing import List, Literal, Optional, Any


class JinaUsage(BaseModel):
    total_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    tokens: Optional[int] = None


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[str] = None

    @field_validator('document', mode='before')
    @classmethod
    def extract_document_text(cls, v: Any) -> Optional[str]:
        if isinstance(v, dict) and 'text' in v:
            return v['text']
        return v


class JinaRerankResponse(BaseModel):
    model: str
    usage: JinaUsage
    results: List[RerankResult]


def jina_rerank_raw(query: str, documents: list[str],
                model: str = "jina-reranker-v2-base-multilingual",
                top_n: Optional[int] = None,
                return_documents: bool = True) -> JinaRerankResponse:
    load_dotenv()
    key = os.getenv("JINA_API_KEY")
    url = 'https://api.jina.ai/v1/rerank'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {key}'
    }
    data = {
        "model": model,
        "query": query,
        "documents": documents,
        "return_documents": return_documents
    }
    if top_n is not None:
        data["top_n"] = top_n

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return JinaRerankResponse.model_validate(response.json())


def jina_rerank(query: str, documents: list[str],
                model: str = "jina-reranker-v2-base-multilingual",
                top_n: Optional[int] = None,
                return_documents: bool = True) -> list[str] | list[RerankResult]:
    response = jina_rerank_raw(query, documents, model, top_n, return_documents)
    return response.results
