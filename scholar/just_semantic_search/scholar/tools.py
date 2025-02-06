from semanticscholar import SemanticScholar


def get_paper_by_title(query: str, limit: int = 20):
    """
    Search for papers by keyword. 
    """
    sch = SemanticScholar()
    results = sch.search_paper(query=query, limit=limit)
    return results

