import os
from eliot import start_action
import typer
from just_semantic_search.meili.rag import EmbeddingModel, MeiliBase, MeiliRAG
from meilisearch_python_sdk.index import SearchResults, Hybrid
import pprint
from typing import Optional, Callable

from meilisearch_python_sdk.models.index import IndexStats


def all_indexes(non_empty: bool = True) -> list[str]:
    """
    Get all indexes that you can use for search.

    Args:
        non_empty (bool): If True, returns only indexes that contain documents, otherwise returns all available indexes. True by default.

    Returns:
        List[str]: A list of index names that can be used for document searches.
    """
    db = MeiliBase()
    return db.non_empty_indexes() if non_empty else db.all_indexes()

def search_documents_raw(query: str, index: str, limit: Optional[int] = 4, semantic_ratio: Optional[float] = 0.5) -> list[dict]:
    """
    Search documents in MeiliSearch database. Giving search results in raw format.
    
    Args:
        query (str): The search query string used to find relevant documents.
        index (str): The name of the index to search within. 
                   It should be one of the allowed list of indexes.
        limit (int): The number of documents to return. 8 by default.

    Returns:
        list[dict]: A list of dictionaries containing the search results.
        Each dictionary contains the following keys:
        - '_rankingScore': The relevance score of the document.
        - '_rankingScoreDetails': A dictionary containing details about the ranking score.
        - 'hash': The unique identifier of the document.
        - 'source': The source document path.
        - 'text': The content of the document.
        - 'token_count': The number of tokens in the document.
        - 'total_fragments': The total number of fragments in the document.

    Example:
        Example result:
        [ {'_rankingScore': 0.718,  # Relevance score of the document
          '_rankingScoreDetails': {'vectorSort': {'order': 0,  # Ranking order
                                                  'similarity': 0.718}},  # Similarity score
          'hash': 'e22c1616...',  # Unique document identifier
          'source': '/path/to/document.txt',  # Source document path
          'text': 'Ageing as a risk factor...',  # Document content
          'token_count': None,  # Number of tokens (if applicable)
          'total_fragments': None},  # Total fragments (if applicable)
          ]
    """
    
    # Get the embedding model from environment variables, defaulting to JINA_EMBEDDINGS_V3
    model_str = os.getenv("EMBEDDING_MODEL", EmbeddingModel.JINA_EMBEDDINGS_V3.value)
    if semantic_ratio is None:
        semantic_ratio = os.getenv("MEILISEARCH_SEMANTIC_RATIO", 0.5)
    
    # Create and return RAG instance with conditional recreate_index
    # It should use default environment variables for host, port, api_key, create_index_if_not_exists, recreate_index
    model = EmbeddingModel(model_str)
    rag = MeiliRAG.get_instance(
        index_name=index,
        model=model,        # The embedding model used for the search
    )
    return rag.search(query, limit=limit, semantic_ratio=semantic_ratio)

def search_documents(query: str, index: str, limit: Optional[int] = 30, semantic_ratio: Optional[float] = 0.5) -> list[str]:
    """
    Search documents in MeiliSearch database.
    
    Args:
        query (str): The search query string used to find relevant documents.
        index (str): The name of the index to search within.
                    It should be one of the allowed list of indexes.
        limit (int): The number of documents to return. 30 by default.
        semantic_ratio (float): The ratio of semantic search. 0.5 by default.
    Returns:
        list[str]: A list of strings containing the search results.
        Each string contains the following keys:
        - '_rankingScore': The relevance score of the document.
        - '_rankingScoreDetails': A dictionary containing details about the ranking score.
        - 'hash': The unique identifier of the document.  
        - 'source': The source document path.
        - 'text': The content of the document.
        - 'token_count': The number of tokens in the document.
        - 'total_fragments': The total number of fragments in the document.

    Example:
        Example result:
        [ {'_rankingScore': 0.718,  # Relevance score of the document
          '_rankingScoreDetails': {'vectorSort': {'order': 0,  # Ranking order
                                                  'similarity': 0.718}},  # Similarity score
          'hash': 'e22c1616...',  # Unique document identifier
          'source': '/path/to/document.txt',  # Source document path
          'text': 'Ageing as a risk factor...',  # Document content
          'token_count': None,  # Number of tokens (if applicable)
          'total_fragments': None},  # Total fragments (if applicable)
          ]
    """
    with start_action(action_type="search_documents", query=query, index=index, limit=limit) as action:
        if semantic_ratio is None:
            semantic_ratio = os.getenv("MEILISEARCH_SEMANTIC_RATIO", 0.5)
            action.log("as semantic ratio, using default value", semantic_ratio=semantic_ratio)
        hits: list[dict] = search_documents_raw(query, index, limit, semantic_ratio=semantic_ratio).hits
        action.log(message_type="search_documents_results_count", count=len(hits))
        result: list[str] = [ h["text"] + "\n SOURCE: " + h["source"] for h in hits]
        return result
    
def search_documents_text(query: str, index: str, limit: Optional[int] = 30) -> list[str]:
    """
    Search documents in MeiliSearch database.
    
    Args:
        query (str): The search query string used to find relevant documents.
        index (str): The name of the index to search within.
                    It should be one of the allowed list of indexes.
        limit (int): The number of documents to return. 30 by default.

    Returns:
        list[str]: A list of strings containing the search results.
        Each string contains the following keys:
        - '_rankingScore': The relevance score of the document.
        - '_rankingScoreDetails': A dictionary containing details about the ranking score.
        - 'hash': The unique identifier of the document.  
        - 'source': The source document path.
        - 'text': The content of the document.
        - 'token_count': The number of tokens in the document.
        - 'total_fragments': The total number of fragments in the document.

    Example:
        Example result:
        [ {'_rankingScore': 0.718,  # Relevance score of the document
          '_rankingScoreDetails': {'vectorSort': {'order': 0,  # Ranking order
                                                  'similarity': 0.718}},  # Similarity score
          'hash': 'e22c1616...',  # Unique document identifier
          'source': '/path/to/document.txt',  # Source document path
          'text': 'Ageing as a risk factor...',  # Document content
          'token_count': None,  # Number of tokens (if applicable)
          'total_fragments': None},  # Total fragments (if applicable)
          ]
    """
    return search_documents(query, index, limit, semantic_ratio=0.0)
    

def search_documents_debug(query: str, index: str, limit: Optional[int] = 30) -> list[dict]:
    """
    Search documents in MeiliSearch database.
    
    Args:
        query (str): The search query string used to find relevant documents.
        index (str): The name of the index to search within. 
                   It should be one of the allowed list of indexes.
        limit (int): The number of documents to return. 30 by default.

    Returns:
        list[dict]: A list of dictionaries containing the search results.
        Each dictionary contains the following keys:
        - '_rankingScore': The relevance score of the document.
        - '_rankingScoreDetails': A dictionary containing details about the ranking score.
        - 'hash': The unique identifier of the document.
        - 'source': The source document path.
        - 'text': The content of the document.
        - 'token_count': The number of tokens in the document.
        - 'total_fragments': The total number of fragments in the document.

    Example:
        Example result:
        [ {'_rankingScore': 0.718,  # Relevance score of the document
          '_rankingScoreDetails': {'vectorSort': {'order': 0,  # Ranking order
                                                  'similarity': 0.718}},  # Similarity score
          'hash': 'e22c1616...',  # Unique document identifier
          'source': '/path/to/document.txt',  # Source document path
          'text': 'Ageing as a risk factor...',  # Document content
          'token_count': None,  # Number of tokens (if applicable)
          'total_fragments': None},  # Total fragments (if applicable)
          ]
    """
    with start_action(action_type="search_documents_debug", query=query, index=index, limit=limit) as action:
        results = search_documents_raw(query, index, limit)
        action.log(message_type="search_documents_debug_results", count=len(results.hits), results=results)
        return [ h["text"] + "\n SOURCE: " + h["source"] for h in results.hits]
