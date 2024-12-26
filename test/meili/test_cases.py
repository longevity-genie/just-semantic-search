from sentence_transformers import SentenceTransformer
from just_semantic_search.embeddings import *
from just_semantic_search.utils.tokens import *
#from just_semantic_search.utils import RenderingFileDestination


from just_semantic_search.meili.rag import *
from just_semantic_search.meili.rag import *
import time

from eliot._output import *
from eliot import start_action

def test_rsids(rag: MeiliRAG, model: Optional[SentenceTransformer] = None) -> SearchResults:

    expected ="""
    In particular for rs123456789 and rs123456788 as well as similar but misspelled rsids are added to the documents:
        * 10.txt contains both two times
        * 11.txt contains both one time
        * 12.txt and 13 contain only one rsid
        * 20.txt contains both wrong rsids two times
        * 21.txt contains both wrong rsids one time
        * 22.txt and 23 contain only one wrong rsid
    """
    
    with start_action(action_type="test_rsids") as action:
        results = rag.search("rs123456789 and rs123456788", model=model)
        texts = [hit["text"] for hit in results.hits]
        sources = [hit["source"] for hit in results.hits]

        for hit in results.hits:
            action.log(f"Type of hit: {type(hit)}")
            
        action.add_success_fields(
            message_type="test_rsids_complete",
            sources=sources,
            texts=texts,
            count=len(results.hits),
            expected=expected
        )
        return results
    
def test_superhero_search(rag: MeiliRAG, model: Optional[SentenceTransformer] = None, tell_text: bool = False):
    expected = """
    Only 114 document has text about superheroes, but text did not contain words 'comics' or 'superheroes'
    """
    with start_action(action_type="test_superhero_search") as action:
        results = rag.search("comic superheroes", model=model)
        texts = [hit["text"] for hit in results.hits]
        sources = [hit["source"] for hit in results.hits]

        action.add_success_fields(
            message_type="test_superhero_search",
            sources=sources,
            texts=texts if tell_text else None,
            count=len(results.hits),
            expected=expected
        )
        return results