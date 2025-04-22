from abc import ABC
from enum import Enum
from sentence_transformers import CrossEncoder
from typing import Optional, Union
from pydantic import BaseModel, Field
from just_semantic_search.remote.jina_reranker import jina_rerank, RerankResult


class RerankingModel(str, Enum):
    JINA_RERANKER_V2_BASE_MULTILINGUAL = "jinaai/jina-reranker-v2-base-multilingual"
    REMOTE_JINA_RERANKER_V2_BASE_MULTILINGUAL = "jinaai/jina-reranker-v2-base-multilingual_remote"
    

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
    

def load_reranker(model: Union[RerankingModel, str]) -> AbstractReranker:
    """
    Loads a CrossEncoder model for reranking tasks.

    Args:
        model: The identifier of the model to load. Can be a RerankingModel enum member
               or a string representing the model name (e.g., from Hugging Face Hub).

    Returns:
        An instance of the CrossEncoder model.
    """
    if model == RerankingModel.REMOTE_JINA_RERANKER_V2_BASE_MULTILINGUAL:
        return RemoteJinaReranker()
    else: 
        model_id = model.value if isinstance(model, RerankingModel) else model
        cross_encoder = CrossEncoder(
            model_id.replace("_remote", ""),
            model_kwargs={
                "torch_dtype": "auto"
            },
            trust_remote_code=True,
        )
        return CrossEncoderReranker(cross_encoder=cross_encoder)
    
    
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

    

class CrossEncoderReranker(AbstractReranker):
    """
    Reranks a list of documents based on their relevance to a given query using a Jina reranker model.
    """
    cross_encoder: Optional[CrossEncoder] = Field(exclude=True)
    return_documents: bool = Field(default=True)
    

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