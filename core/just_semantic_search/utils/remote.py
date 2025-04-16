import os
from dotenv import load_dotenv
import requests
from pydantic import BaseModel, field_validator
from typing import List, Literal, Optional, Any
from enum import Enum


class JinaTask(str, Enum):
    QUERY = "retrieval.query"
    PASSAGE = "retrieval.passage"
    SEPARATION = "separation"
    CLASSIFICATION = "classification"
    TEXT_MATCHING = "text-matching"


class JinaUsage(BaseModel):
    total_tokens: int
    prompt_tokens: Optional[int] = None # rerank usage doesn't have prompt_tokens


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

class JinaRerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[str] = None # Changed type back to str

    @field_validator('document', mode='before')
    @classmethod
    def extract_document_text(cls, v: Any) -> Optional[str]:
        if isinstance(v, dict) and 'text' in v:
            return v['text']
        # Handle cases where document is not returned or is already a string (unlikely)
        return v

class JinaRerankResponse(BaseModel):
    model: str
    usage: JinaUsage
    results: List[JinaRerankResult]


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


def jina_rerank_raw(query: str, documents: list[str],
                model: str = "jina-reranker-v2-base-multilingual",
                top_n: Optional[int] = None,
                return_documents: bool = True) -> JinaRerankResponse: # Changed return type
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
        "top_n": top_n,
        "documents": documents,
        "return_documents": return_documents
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status() # Added status check
    return JinaRerankResponse.model_validate(response.json()) # Parse with Pydantic


def jina_rerank(query: str, documents: list[str],
                model: str = "jina-reranker-v2-base-multilingual",
                top_n: Optional[int] = None,
                return_documents: bool = True) -> list[str] | list[JinaRerankResult]: # Return type depends on return_documents
    response = jina_rerank_raw(query, documents, model, top_n, return_documents)

    if return_documents:
        # Access the document string directly
        return [result.document for result in response.results if result.document is not None]
    else:
        # Return the results list containing index and score if documents are not requested
        return response.results


if __name__ == "__main__":
    documents = [
         "Organic skincare for sensitive skin with aloe vera and chamomile: Imagine the soothing embrace of nature with our organic skincare range, crafted specifically for sensitive skin. Infused with the calming properties of aloe vera and chamomile, each product provides gentle nourishment and protection. Say goodbye to irritation and hello to a glowing, healthy complexion.",
        "New makeup trends focus on bold colors and innovative techniques: Step into the world of cutting-edge beauty with this seasons makeup trends. Bold, vibrant colors and groundbreaking techniques are redefining the art of makeup. From neon eyeliners to holographic highlighters, unleash your creativity and make a statement with every look.",
        "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille: Erleben Sie die wohltuende Wirkung unserer Bio-Hautpflege, speziell für empfindliche Haut entwickelt. Mit den beruhigenden Eigenschaften von Aloe Vera und Kamille pflegen und schützen unsere Produkte Ihre Haut auf natürliche Weise. Verabschieden Sie sich von Hautirritationen und genießen Sie einen strahlenden Teint.",
        "Neue Make-up-Trends setzen auf kräftige Farben und innovative Techniken: Tauchen Sie ein in die Welt der modernen Schönheit mit den neuesten Make-up-Trends. Kräftige, lebendige Farben und innovative Techniken setzen neue Maßstäbe. Von auffälligen Eyelinern bis hin zu holografischen Highlightern – lassen Sie Ihrer Kreativität freien Lauf und setzen Sie jedes Mal ein Statement.",
        "Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla: Descubre el poder de la naturaleza con nuestra línea de cuidado de la piel orgánico, diseñada especialmente para pieles sensibles. Enriquecidos con aloe vera y manzanilla, estos productos ofrecen una hidratación y protección suave. Despídete de las irritaciones y saluda a una piel radiante y saludable.",
        "Las nuevas tendencias de maquillaje se centran en colores vivos y técnicas innovadoras: Entra en el fascinante mundo del maquillaje con las tendencias más actuales. Colores vivos y técnicas innovadoras están revolucionando el arte del maquillaje. Desde delineadores neón hasta iluminadores holográficos, desata tu creatividad y destaca en cada look.",
        "针对敏感肌专门设计的天然有机护肤产品：体验由芦荟和洋甘菊提取物带来的自然呵护。我们的护肤产品特别为敏感肌设计，温和滋润，保护您的肌肤不受刺激。让您的肌肤告别不适，迎来健康光彩。",
        "新的化妆趋势注重鲜艳的颜色和创新的技巧：进入化妆艺术的新纪元，本季的化妆趋势以大胆的颜色和创新的技巧为主。无论是霓虹眼线还是全息高光，每一款妆容都能让您脱颖而出，展现独特魅力。",
        "敏感肌のために特別に設計された天然有機スキンケア製品: アロエベラとカモミールのやさしい力で、自然の抱擁を感じてください。敏感肌用に特別に設計された私たちのスキンケア製品は、肌に優しく栄養を与え、保護します。肌トラブルにさようなら、輝く健康な肌にこんにちは。",
        "新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています: 今シーズンのメイクアップトレンドは、大胆な色彩と革新的な技術に注目しています。ネオンアイライナーからホログラフィックハイライターまで、クリエイティビティを解き放ち、毎回ユニークなルックを演出しましょう。"
    ]
    query = "Organic skincare products for sensitive skin"
    print("=======================================================")
    # Example with documents returned (default)
    reranked_response = jina_rerank_raw(query, documents)
    for result in reranked_response.results:
        print(result)
