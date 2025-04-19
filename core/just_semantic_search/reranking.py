from abc import ABC
from enum import Enum
from sentence_transformers import CrossEncoder
from typing import Optional, Union
from pydantic import BaseModel, Field
from just_semantic_search.utils.remote import jina_rerank, RerankResult



class RerankingModel(str, Enum):
    JINA_RERANKER_V2_BASE_MULTILINGUAL = "jinaai/jina-reranker-v2-base-multilingual"
    

def load_reranking_model(model: Union[RerankingModel, str]) -> CrossEncoder:
    """
    Loads a CrossEncoder model for reranking tasks.

    Args:
        model: The identifier of the model to load. Can be a RerankingModel enum member
               or a string representing the model name (e.g., from Hugging Face Hub).

    Returns:
        An instance of the CrossEncoder model.
    """
    model_id = model.value if isinstance(model, RerankingModel) else model
    return CrossEncoder(
        model_id,
        model_kwargs={
            "torch_dtype": "auto"
        },
        trust_remote_code=True,
    )

class AbstractReranker(BaseModel, ABC):
    """
    Abstract base class for reranking models.
    """
    convert_to_tensor: bool = Field(default=False)
    
    model_config = {
        "arbitrary_types_allowed": True,
    }
    
    def score(self, query: str, documents: list[str]) -> list[float]:
        """
        Scores a list of documents based on their relevance to a given query.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def rank(self, query: str, documents: list[str], top_k: Optional[int] = None) -> list[str]:
        """
        Reranks a list of documents based on their relevance to a given query.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
class RemoteJinaReranker(AbstractReranker):
    """
    Reranks a list of documents based on their relevance to a given query using a Jina reranker model.
    """
    model: RerankingModel = Field(default=RerankingModel.JINA_RERANKER_V2_BASE_MULTILINGUAL)
    return_documents: bool = Field(default=True)
    
    def model_post_init(self, __context):
        pass  # No need to load a model for remote reranking

    def score(self, query: str, documents: list[str], top_n: Optional[int]=None) -> list[float]:
        """
        Calculates similarity scores between a query and a list of documents using a CrossEncoder model.

        If no model is provided, it defaults to loading the JINA_RERANKER_V2_BASE_MULTILINGUAL model.

        Args:
            query: The search query string.
            documents: A list of document strings to be scored against the query.

        Returns:
            A list of float scores representing the similarity between the query and each document.
        """
        
        # Get raw results with scores
        results = jina_rerank(query, documents, return_documents=self.return_documents, top_n=top_n)
        # Extract just the scores
        return [result.relevance_score for result in results]

    def rank(self, query: str, documents: list[str], top_n: Optional[int] = None) -> list[str]:
        """
        Reranks a list of documents based on their relevance to a given query using a Jina reranker model.

        If no model is provided, it defaults to loading the JINA_RERANKER_V2_BASE_MULTILINGUAL model.

        Args:
            query: The search query string.
            documents: A list of document strings to be reranked.
            top_k: Optional maximum number of documents to return. If None, all documents are returned.

        Returns:
            A list containing the reranked results. The format depends on the `return_documents` parameter.
            If `return_documents` is True, each item is a dictionary with 'corpus_id', 'score', and 'text'.
            If `return_documents` is False, it's a list of scores.
        """
        rankings = jina_rerank(query, documents, return_documents=self.return_documents, top_n=top_n)
        return rankings

    

class Reranker(AbstractReranker):
    """
    Reranks a list of documents based on their relevance to a given query using a Jina reranker model.
    """
    model: RerankingModel = Field(default=RerankingModel.JINA_RERANKER_V2_BASE_MULTILINGUAL)
    cross_encoder: Optional[CrossEncoder] = Field(default=None)
    return_documents: bool = Field(default=True)
    
    def model_post_init(self, __context):
        if self.cross_encoder is None:
            self.cross_encoder = load_reranking_model(self.model)

    def score(self, query: str, documents: list[str]) -> list[float]:
        """
        Calculates similarity scores between a query and a list of documents using a CrossEncoder model.

        If no model is provided, it defaults to loading the JINA_RERANKER_V2_BASE_MULTILINGUAL model.

        Args:
            query: The search query string.
            documents: A list of document strings to be scored against the query.
            convert_to_tensor: Whether to convert the output scores to PyTorch tensors before converting to list. Defaults to False.
            model: An optional pre-loaded CrossEncoder model instance. If None, the default
                Jina multilingual model is loaded.

        Returns:
            A list of float scores representing the similarity between the query and each document.
        """
        sentence_pairs = [[query, doc] for doc in documents]
        scores = self.cross_encoder.predict(sentence_pairs, convert_to_tensor=self.convert_to_tensor).tolist()
        return scores

    def rank(self, query: str, documents: list[str], top_n: Optional[int] = None) -> list[str]:
        """
        Reranks a list of documents based on their relevance to a given query using a Jina reranker model.

        If no model is provided, it defaults to loading the JINA_RERANKER_V2_BASE_MULTILINGUAL model.

        Args:
            query: The search query string.
            documents: A list of document strings to be reranked.
            convert_to_tensor: Whether to convert the output scores to PyTorch tensors. Defaults to False.
            return_documents: If True (default), returns a list of dictionaries, each containing 'corpus_id',
                            'score', and 'text'. If False, returns only the scores.
            model: An optional pre-loaded CrossEncoder model instance. If None, the default
                Jina multilingual model is loaded.

        Returns:
            A list containing the reranked results. The format depends on the `return_documents` parameter.
            If `return_documents` is True, each item is a dictionary with 'corpus_id', 'score', and 'text'.
            If `return_documents` is False, it's a list of scores.
        """
        rankings = self.cross_encoder.rank(query, documents, return_documents=self.return_documents, convert_to_tensor=self.convert_to_tensor, top_k=top_n)
        return [RerankResult(index=result["corpus_id"], relevance_score=result["score"], document=result["text"]) for result in rankings]



if __name__ == "__main__":

    convert_to_tensor = False
    local_reranker = Reranker(
        model=RerankingModel.JINA_RERANKER_V2_BASE_MULTILINGUAL,
        convert_to_tensor=convert_to_tensor
    )

    remote_reranker = RemoteJinaReranker(
        model=RerankingModel.JINA_RERANKER_V2_BASE_MULTILINGUAL,
        convert_to_tensor=convert_to_tensor,
        return_documents=True
    )
    
    
    # Example query and documents
    query = "Organic skincare products for sensitive skin"
    documents = [
        "Organic skincare for sensitive skin with aloe vera and chamomile.",
        "New makeup trends focus on bold colors and innovative techniques",
        "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille",
        "Neue Make-up-Trends setzen auf kräftige Farben und innovative Techniken",
        "Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla",
        "Las nuevas tendencias de maquillaje se centran en colores vivos y técnicas innovadoras",
        "针对敏感肌专门设计的天然有机护肤产品",
        "新的化妆趋势注重鲜艳的颜色和创新的技巧",
        "敏感肌のために特別に設計された天然有機スキンケア製品",
        "新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています",
    ]
    
    # Score documents
    #scores = local_reranker.score(query, documents)
    #print(scores)
    #scores_2 = remote_reranker.score(query, documents)
    #print(scores_2)
    rankings = local_reranker.rank(query, documents)
    print(f"Query: {query}")
    for ranking in rankings:
        print(ranking)
    
    print("=======================================================")
    rankings = remote_reranker.rank(query, documents)
    for ranking in rankings:
        print(ranking)
    