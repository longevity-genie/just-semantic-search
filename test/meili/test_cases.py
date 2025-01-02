import rich
from sentence_transformers import SentenceTransformer
from just_semantic_search.embeddings import *
from just_semantic_search.utils.tokens import *
#from just_semantic_search.utils import RenderingFileDestination


from just_semantic_search.meili.rag import *
from just_semantic_search.meili.rag import *
import time
from rich.pretty import pprint

from eliot._output import *
from eliot import start_action

def test_rsids(rag: MeiliRAG, model: Optional[SentenceTransformer] = None, tell_text: bool = False, score_threshold: float = 0.75) -> SearchResults:

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
        results: SearchResults = rag.search("rs123456789 and rs123456788", model=model)
        hits = [hit for hit in results.hits if hit["_rankingScore"] >= score_threshold]
        texts = [hit["text"] for hit in hits]
        sources = [hit["source"] for hit in hits]
        scores = [hit["_rankingScore"] for hit in hits]
        print('first hit:')
        pprint(results.hits[0])
        scored_sources = [{"source": source, "score": score} for source, score in zip(sources, scores)]
            
        action.add_success_fields(
            message_type="test_rsids_complete",
            sources=scored_sources,
            texts=texts if tell_text else None,
            count=len(hits),
            expected=expected,
            semantic_hit_count=results.semantic_hit_count,
            score_threshold=score_threshold
        )
        return results
    
def test_superhero_search(rag: MeiliRAG, model: Optional[SentenceTransformer] = None, tell_text: bool = False, score_threshold: float = 0.75):
    expected = """
    Only 114 document has text about superheroes, but text did not contain words 'comics' or 'superheroes'
    """
    with start_action(action_type="test_superhero_search") as action:
        results = rag.search("comic superheroes", model=model)
      
        hits = [hit for hit in results.hits if hit["_rankingScore"] >= score_threshold]
        texts = [hit["text"] for hit in hits]
        sources = [hit["source"] for hit in hits]
        scores = [hit["_rankingScore"] for hit in hits]
        
        print('first hit:')
        pprint(results.hits[0])
        scored_sources = [{"source": source, "score": score} for source, score in zip(sources, scores)]
        
        action.add_success_fields(
            message_type="test_superhero_search",
            sources=scored_sources,
            texts=texts if tell_text else None,
            count=len(hits),
            expected=expected,
            semantic_hit_count=results.semantic_hit_count,
        )
        return results