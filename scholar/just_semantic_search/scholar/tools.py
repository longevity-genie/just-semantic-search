from semanticscholar import SemanticScholar
import os
from rich.pretty import pprint

from semanticscholar.PaginatedResults import PaginatedResults

def search_papers(query: str, limit: int = 20):
    """
    Search for papers by keyword. 
    """
    # Retrieve the API key from the environment:
    # 1. Check for "SEMANTIC_SCHOLAR_API_KEY"
    # 2. Otherwise, check for "SEMANTIC_SCHOLAR_API"
    # 3. If neither is set, the API key will be None.
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("SEMANTIC_SCHOLAR_API")
    sch = SemanticScholar(api_key=api_key)
    results = sch.search_paper(query=query, limit=limit)
    return results


def get_scholar_paper_by_id(paper_id: str):
    """
    Get a paper by its ID.
    """
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("SEMANTIC_SCHOLAR_API")
    sch = SemanticScholar()
    paper = sch.paper(paper_id)
    return paper