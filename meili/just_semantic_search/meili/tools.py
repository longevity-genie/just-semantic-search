import os
from eliot import start_action
from just_semantic_search.meili.rag import EmbeddingModel, MeiliBase, MeiliRAG
from typing import Optional
from meilisearch_python_sdk.index import SearchResults


def all_indexes(non_empty: bool = True, debug: bool = True) -> list[str]:
    """
    Get all indexes that you can use for search.

    Args:
        non_empty (bool): If True, returns only indexes that contain documents, otherwise returns all available indexes. True by default.

    Returns:
        List[str]: A list of index names that can be used for document searches.
    """
    host = os.getenv("MEILISEARCH_HOST", "127.0.0.1")
    port = os.getenv("MEILISEARCH_PORT", 7700)
    api_key = os.getenv("MEILISEARCH_API_KEY", "fancy_master_key")
    if debug:
        with start_action(action_type="all_indexes", host=host, port=port) as action:
            db = MeiliBase(host=host, port=port, api_key=api_key)
            action.log(message_type="all_indexes", count=len(db.all_indexes()))
            return db.non_empty_indexes() if non_empty else db.all_indexes()
    else:
        db = MeiliBase(host=host, port=port, api_key=api_key)
        return db.non_empty_indexes() if non_empty else db.all_indexes()


def search_documents_raw(query: str, index: str, limit: Optional[int] = 8, semantic_ratio: Optional[float] = 0.5, debug: bool = True) -> SearchResults:
    """
    Search documents in MeiliSearch database. Giving search results in raw format.
    
    Args:
        query (str): The search query string used to find relevant documents.
        index (str): The name of the index to search within. 
                   It should be one of the allowed list of indexes.
        limit (int): The number of documents to return. 8 by default.
        semantic_ratio (float): The ratio of semantic search to keyword search. 0.5 by default.
        debug (bool): If True, logs debugging information. True by default.

    Returns:
        SearchResults: The MeiliSearch results object containing hits and search metadata.
        Each hit typically contains:
        - '_rankingScore': The relevance score of the document.
        - '_rankingScoreDetails': A dictionary containing details about the ranking score.
        - 'hash': The unique identifier of the document.
        - 'source': The source document path.
        - 'text': The content of the document.
        - 'token_count': The number of tokens in the document.
        - 'total_fragments': The total number of fragments in the document.
    """
    if semantic_ratio is None:
        semantic_ratio = os.getenv("MEILISEARCH_SEMANTIC_RATIO", 0.5)    
    host = os.getenv("MEILISEARCH_HOST", "127.0.0.1")
    port = os.getenv("MEILISEARCH_PORT", 7700)
    api_key = os.getenv("MEILISEARCH_API_KEY", "fancy_master_key")
    model_str = os.getenv("EMBEDDING_MODEL", EmbeddingModel.JINA_EMBEDDINGS_V3.value)
    model = EmbeddingModel(model_str)
    rag = MeiliRAG.get_instance(
        host=host,
        port=port,
        api_key=api_key,
        index_name=index,
        model=model,        # The embedding model used for the search
    )
    if debug:
        with start_action(action_type="search_documents", query=query, index=index, limit=limit) as action:    
            action.log(message_type="search_documents", host=host, port=port, model_str=model_str, semantic_ratio=semantic_ratio, index=index)
            # Create and return RAG instance with conditional recreate_index
            # It should use default environment variables for host, port, api_key, create_index_if_not_exists, recreate_index
            result = rag.search(query, limit=limit, semantic_ratio=semantic_ratio)
            hits: list[dict] = result.hits
            action.log(message_type="search_documents_results_count", count=len(hits))
            return result
    else:
        result = rag.search(query, limit=limit, semantic_ratio=semantic_ratio)
        return result 

def search_documents(query: str, index: str, limit: Optional[int] = 8, semantic_ratio: Optional[float] = 0.5, debug: bool = True) -> list[str]:
    """
    Search documents in MeiliSearch database.
    
    Args:
        query (str): The search query string used to find relevant documents.
        index (str): The name of the index to search within.
                    It should be one of the allowed list of indexes.
        limit (int): The number of documents to return. 8 by default.
        semantic_ratio (float): The ratio of semantic search. 0.5 by default.
        debug (bool): If True, print debug information. True by default.
    Returns:
        list[str]: A list of strings containing the document text followed by the source.
        Each string contains the document content and its source separated by "\n SOURCE: ".

    Example:
        Example result:
        [
            "Ageing as a risk factor...\n SOURCE: /path/to/document.txt",
            "Another document content...\n SOURCE: /path/to/another/document.txt"
        ]
    """
    result = []
    for h in search_documents_raw(query, index, limit, semantic_ratio=semantic_ratio, debug=debug).hits:
        doc_info = h["text"]
        # Add title if it exists
        if "title" in h:
            doc_info = f"Title: {h['title']}\n{doc_info}"
        # Add fragment information
        if "fragment_num" in h and "total_fragments" in h:
            doc_info += f"\nFragment: {h['fragment_num']} out of {h['total_fragments']}"
        if "token_count" in h:
            doc_info += f"\nToken count: {h['token_count']}"
        # Add source
        doc_info += f"\nSOURCE: {h['source']}"
        result.append(doc_info)
    return result
    
def search_documents_text(query: str, index: str, limit: Optional[int] = 8, debug: bool = True) -> list[str]:
    """
    Search documents in MeiliSearch database using only text/keyword search (no semantic search).
    
    Args:
        query (str): The search query string used to find relevant documents.
        index (str): The name of the index to search within.
                    It should be one of the allowed list of indexes.
        limit (int): The number of documents to return. 8 by default.
        debug (bool): If True, print debug information. True by default.

    Returns:
        list[str]: A list of strings containing the search results which also mention the source of the document.

    Example:
        Example result:
        [ " Built upon scDiffCom, scAgeCom is an atlas of age-related cell-cell communication changes covering 23 mouse tissues from 58 single-cell RNA sequencing datasets from Tabula Muris Senis and the Calico murine aging cell atlas... \n SOURCE:https://doi.org/10.1038/s43587-023-00514-x",
          "Deep learning models achieve state-of-the art results in predicting blood glucose trajectories, with a wide range of architectures being proposed....\n SOURCE:https://arxiv.org/abs/2209.04526",
          ]
    """
    return search_documents(query, index, limit, semantic_ratio=0.0, debug=debug)