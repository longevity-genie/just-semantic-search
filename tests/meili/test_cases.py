from just_semantic_search.splitter_factory import SplitterType
import pytest
from just_semantic_search.meili.rag import MeiliRAG, SearchResults
from just_semantic_search.meili.utils.services import ensure_meili_is_running
from eliot import start_action
from tests.config import *
from just_semantic_search.embeddings import EmbeddingModel, load_sentence_transformer_from_enum
from pycomfort.logging import to_nice_stdout
from tests.meili.functions import index_file, simulate_meilisearch_disconnection

to_nice_stdout()


@pytest.fixture
def model() -> EmbeddingModel:
    return EmbeddingModel.JINA_EMBEDDINGS_V3


@pytest.fixture
def rag(request, model: EmbeddingModel) -> MeiliRAG:
    host = "127.0.0.1"
    port = 7700
    api_key = "fancy_master_key"
    
    # Set default parameters if not parametrized
    if hasattr(request, 'param'):
        recreate_index, index_name, populate = request.param
    else:
        recreate_index, index_name, populate = False, "robi-tacutu", True
    
    
    # Create and return RAG instance with conditional recreate_index
    rag = MeiliRAG(
        index_name=index_name,
        model=model,
        host=host,
        port=port,
        api_key=api_key,
        create_index_if_not_exists=True,
        recreate_index=recreate_index,
        init_callback=lambda rag: ensure_meili_is_running(host=rag.host, port=rag.port)
    )
    stats = rag.index.get_stats()
    with start_action(action_type="index_population_check") as action:
        if stats.number_of_documents == 0 and populate:
            action.log(
                message_type="index_population",
                index_name=index_name,
                message=f"{index_name} index is empty, filling it with the data"
            )
            rag.index_folder(tacutopapers_dir)

    return rag


def test_rsids(rag: MeiliRAG, tell_text: bool = False, score_threshold: float = 0.75) -> SearchResults:

    transformer_model = load_sentence_transformer_from_enum(rag.model)
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
        results: SearchResults = rag.search("rs123456789 and rs123456788", model=transformer_model)
        hits = [hit for hit in results.hits if hit["_rankingScore"] >= score_threshold]
        texts = [hit["text"] for hit in hits]
        sources = [hit["source"] for hit in hits]
        sources_last_part = [source.split("/")[-1] for source in sources]

        scores = [hit["_rankingScore"] for hit in hits]
        fields = results.hits[0].keys()
        action.log(
            message_type="first_hit_fields",
            fields=fields
        )
        action.log(
            message_type="first_hit_source",
            source=results.hits[0]["source"]
        )
        scored_sources = [{"source": source, "score": score} for source, score in zip(sources, scores)]
        assert "10.txt" in sources_last_part[:2] , "10.txt contains both two times"
        assert "11.txt" in sources_last_part[:2] , "11.txt contains both two times"
        assert "12.txt" in sources_last_part[3:5] , "12.txt contains only one rsid"
            
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
    

def test_superhero_search(rag: MeiliRAG, tell_text: bool = False, score_threshold: float = 0.75) -> SearchResults:
    transformer_model = load_sentence_transformer_from_enum(rag.model)
    with start_action(action_type="test_superhero_search") as action:
        results = rag.search("comic superheroes", model=transformer_model)
      
        hits = [hit for hit in results.hits if hit["_rankingScore"] >= score_threshold]
        texts = [hit["text"] for hit in hits]
        sources = [hit["source"] for hit in hits]
        sources_last_part = [source.split("/")[-1] for source in sources]
        scores = [hit["_rankingScore"] for hit in hits]
        
        action.log('first hit:')
        action.log(str(results.hits[0]))
        scored_sources = [{"source": source, "score": score} for source, score in zip(sources, scores)]
        assert "114.txt" in sources_last_part[0], "Only 114 document has text about superheroes, but text did not contain words 'comics' or 'superheroes'"
        
        action.add_success_fields(
            message_type="test_superhero_search",
            sources=scored_sources,
            texts=texts if tell_text else None,
            count=len(hits),
            semantic_hit_count=results.semantic_hit_count,
        )
        return results
    

@pytest.mark.parametrize('rag', [(True, "test", False)], indirect=True)
def test_retry(rag: MeiliRAG) -> SearchResults:
      
    abstract: str = "Multiple studies characterizing the human ageing phenotype..."
    title: str = "The Digital Ageing Atlas: integrating the diversity of age-related changes into a unified resource"
    source: str = "https://doi.org/10.1093/nar/gku843"
    splitter_type: SplitterType = SplitterType.ARTICLE

    file_1 = tacutopapers_dir / "108.txt"
    
    simulate_meilisearch_disconnection(duration=10)
    index_file(rag, file_1, abstract, title, source, splitter_type)
    
    docs = list(rag.get_documents().results)
    print(f"Loaded docs: {len(docs)}")

    assert any("the human aging-related gene set presents" in doc["text"] for doc in docs), "No element satisfies the condition"


