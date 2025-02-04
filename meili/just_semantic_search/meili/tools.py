import os
from eliot import start_action
import typer
from just_semantic_search.meili.rag import EmbeddingModel, MeiliBase, MeiliRAG
from meilisearch_python_sdk.index import SearchResults, Hybrid
import pprint
from typing import Optional, Callable

from meilisearch_python_sdk.models.index import IndexStats

def all_indexes() -> list[str]:
    """
    Get all indexes that you can use for search
    """
    db = MeiliBase()
    return db.non_empty_indexes()

def search_documents(query: str, index: str, limit: Optional[int] = 4) -> list[dict]:
    """
    Search documents in MeiliSearch database.
    
    Parameters:
    - query (str): The search query string used to find relevant documents.
    - index (str): The name of the index to search within. 
                   It should be one of the allowed list of indexes.
    - limit (int): The number of documents to return. 8 by default.

    Example of result:
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
    model = EmbeddingModel(model_str)
    
    # Create and return RAG instance with conditional recreate_index
    # It should use default environment variables for host, port, api_key, create_index_if_not_exists, recreate_index
    rag = MeiliRAG(
        index_name=index,
        model=model,        # The embedding model used for the search
    )
    return rag.search(query, limit=limit).hits