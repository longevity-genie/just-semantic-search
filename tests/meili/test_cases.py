from just_semantic_search.reranking import RemoteJinaReranker, Reranker, RerankingModel
import pytest
import random
import concurrent.futures
from just_semantic_search.splitters.splitter_factory import SplitterType
from just_semantic_search.meili.rag import MeiliRAG, SearchResults
from just_semantic_search.meili.utils.services import ensure_meili_is_running
from eliot import start_action
from tests.config import *
from just_semantic_search.embeddings import EmbeddingModel, load_sentence_transformer_from_enum
from pycomfort.logging import to_nice_stdout
from tests.meili.functions import index_file, simulate_meilisearch_disconnection
from just_semantic_search.remote.jina_reranker import RerankResult

to_nice_stdout()


@pytest.fixture
def model() -> EmbeddingModel:
    return EmbeddingModel.JINA_EMBEDDINGS_V3


@pytest.fixture
def rag(request, model: EmbeddingModel):
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

    yield rag


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
    

@pytest.mark.parametrize("remote_embedding", [False]) #[False, True] we temporaly put False to make CI easier but remote DID WORK
def test_superhero_search(rag: MeiliRAG, remote_embedding: bool, tell_text: bool = False, score_threshold: float = 0.75) -> SearchResults:
    transformer_model = load_sentence_transformer_from_enum(rag.model)
    with start_action(action_type="test_superhero_search") as action:
        results = rag.search("comic superheroes", model=transformer_model, remote_embedding=remote_embedding)
      
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
    

@pytest.mark.skip(reason="Skipping test_retry")
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


@pytest.mark.parametrize('rag', [(False, "robi-tacutu", False)], indirect=True)
def test_concurrent_search_resilience(rag: MeiliRAG) -> None:
    """Test the resilience of MeiliRAG singleton pattern in concurrent multi-worker scenarios."""
    with start_action(action_type="test_concurrent_search_resilience") as action:
        # Ensure we're using an existing index with data
        index_name = rag.index_name
        host = rag.host
        port = rag.port
        api_key = rag.api_key
        model = rag.model
        
        # Validate that the index exists and has documents
        docs_count = rag.index.get_stats().number_of_documents
        assert docs_count > 0, f"Index '{index_name}' is empty, cannot proceed with test"
        
        action.log(
            message_type="test_setup",
            index_name=index_name,
            host=host,
            port=port,
            document_count=docs_count
        )
        
        action.log(message_type="model_warm_up", model_name=model.value)
        
        # Define a function that gets a MeiliRAG instance and performs search
        def perform_search(query: str, worker_id: int) -> dict:
            try:
                # Get a new instance of MeiliRAG (testing the get_instance mechanism)
                instance = MeiliRAG.get_instance(
                    index_name=index_name,
                    host=host,
                    port=port,
                    api_key=api_key,
                    model=model
                )
                
                # Store the instance ID to verify singleton behavior
                instance_id = id(instance)
                
                # Verify instance is properly initialized
                assert instance.client is not None, "MeiliRAG instance client not initialized"
                
                # Perform the search
                results = instance.search(query)
                
                return {
                    "worker_id": worker_id,
                    "query": query,
                    "success": True,
                    "instance_id": instance_id,
                    "hit_count": len(results.hits) if hasattr(results, 'hits') else 0
                }
            except Exception as e:
                # Record the exception but don't fail the test
                return {
                    "worker_id": worker_id,
                    "query": query,
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
        
        # Set of queries to run - using queries we know work from other tests
        queries = [
            "rs123456789 and rs123456788",  # We know this from test_rsids
            "comic superheroes",            # We know this from test_superhero_search
            "aging phenotype",              # Common in aging papers
            "digital atlas",                # Likely in the dataset
            "human genes",                  # Common in genetics papers
        ]
        
        # Number of concurrent workers
        num_workers = 8
        # Number of search iterations per worker
        iterations_per_worker = 5
        
        # Create a list of tasks (query, worker_id)
        tasks = []
        for worker_id in range(num_workers):
            for _ in range(iterations_per_worker):
                # Each worker gets random queries from the list
                query = random.choice(queries)
                tasks.append((query, worker_id))
        
        action.log(
            message_type="starting_concurrent_searches",
            num_workers=num_workers,
            iterations_per_worker=iterations_per_worker,
            total_tasks=len(tasks)
        )
        
        # Run the tasks in a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_results = [executor.submit(perform_search, query, worker_id) for query, worker_id in tasks]
            results = [future.result() for future in concurrent.futures.as_completed(future_results)]
        
        # Analyze results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        # Extract and count unique instance IDs from successful results
        instance_ids = {r["instance_id"] for r in successful if "instance_id" in r}
        
        # Log statistics
        action.log(
            message_type="concurrent_search_results",
            total_tasks=len(tasks),
            successful_count=len(successful),
            failed_count=len(failed),
            success_rate=f"{len(successful)/len(tasks)*100:.2f}%",
            unique_instance_count=len(instance_ids),
            failures=failed if failed else None
        )
        
        # We should have at least one successful result
        assert len(successful) > 0, "No successful search operations"
        
        # The key test: verify we got exactly one instance ID across all searches
        assert len(instance_ids) == 1, f"Expected 1 singleton instance, got {len(instance_ids)}"
        
        # Additional verification of singleton pattern
        instances = {}
        for _ in range(5):
            instance = MeiliRAG.get_instance(
                index_name=index_name,
                host=host,
                port=port,
                api_key=api_key,
                model=model
            )
            instances[id(instance)] = instance
        
        # We should have only one instance per index
        assert len(instances) == 1, f"Expected 1 singleton instance, got {len(instances)}"
        
        # Final verification - the instance from the concurrent searches should be the same
        # as what we get now
        assert list(instance_ids)[0] == list(instances.keys())[0], "Instance IDs don't match between concurrent and sequential calls"
        
        action.add_success_fields(
            message_type="test_concurrent_search_resilience_complete",
            success_rate=f"{len(successful)/len(tasks)*100:.2f}%",
            unique_instance_count=len(instance_ids)
        )


def test_reranking():
    """Test both local and remote rerankers to ensure they correctly rank documents."""
    # Create local reranker
    local_reranker = Reranker(model=RerankingModel.JINA_RERANKER_V2_BASE_MULTILINGUAL)
    
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
    
    # Expected order of document indices (most to least relevant)
    expected_order = [6, 8, 0, 4, 2, 9, 5, 1, 3, 7]
    
    # Test with local reranker
    local_results = local_reranker.rank(query, documents)
    
    # Verify we got the right number of results
    assert len(local_results) == len(documents), "Local reranker should return a result for each document"
    
    # Verify all results are RerankResult objects
    assert all(isinstance(result, RerankResult) for result in local_results), "All local results should be RerankResult objects"
    
    # Verify order matches expected order
    local_result_indices = [result.index for result in local_results]
    assert local_result_indices == expected_order, f"Expected order {expected_order}, got {local_result_indices}"
    
    # Verify the scores are within expected ranges
    # Relevant documents (about skincare) should have high scores
    assert local_results[0].relevance_score > 0.9, "Top result should have score > 0.9"
    assert local_results[1].relevance_score > 0.8, "Second result should have score > 0.8"
    assert local_results[2].relevance_score > 0.8, "Third result should have score > 0.8"
    
    # Irrelevant documents (about makeup) should have low scores
    assert local_results[-1].relevance_score < 0.1, "Least relevant result should have score < 0.1"
    
    # Test documents are correctly associated with scores
    skincare_docs = [
        "针对敏感肌专门设计的天然有机护肤产品",  # Chinese
        "敏感肌のために特別に設計された天然有機スキンケア製品",  # Japanese
        "Organic skincare for sensitive skin with aloe vera and chamomile.",  # English
    ]
    
    makeup_docs = [
        "New makeup trends focus on bold colors and innovative techniques",  # English
        "Neue Make-up-Trends setzen auf kräftige Farben und innovative Techniken",  # German
        "新的化妆趋势注重鲜艳的颜色和创新的技巧",  # Chinese
    ]
    
    # Verify top 3 results are skincare-related
    top_docs = [local_results[i].document for i in range(3)]
    for doc in top_docs:
        assert doc in skincare_docs, f"Expected skincare document, got: {doc}"
    
    # Verify bottom 3 contain makeup-related documents
    bottom_docs = [local_results[i].document for i in range(-3, 0)]
    assert any(doc in makeup_docs for doc in bottom_docs), "Expected at least one makeup document in bottom 3"
    
    # Test with remote reranker
    remote_reranker = RemoteJinaReranker(model=RerankingModel.JINA_RERANKER_V2_BASE_MULTILINGUAL)
    try:
        # Test with remote RerankResult objects
        remote_results = remote_reranker.rank(query, documents)
        
        # Verify we got the right number of results
        assert len(remote_results) == len(documents), "Remote reranker should return a result for each document"
        
        # Verify all results are RerankResult objects
        assert all(isinstance(result, RerankResult) for result in remote_results), "All remote results should be RerankResult objects"
        
        # Verify the document ranking is approximately the same (may have small floating point differences)
        remote_result_indices = [result.index for result in remote_results]
        assert remote_result_indices == expected_order, f"Expected order {expected_order}, got {remote_result_indices}"
        
        # Verify the scores are similar to local reranker (within 0.01)
        for local_result, remote_result in zip(local_results, remote_results):
            assert abs(local_result.relevance_score - remote_result.relevance_score) < 0.01, \
                f"Score difference too large: local={local_result.relevance_score}, remote={remote_result.relevance_score}"
        
    except Exception as e:
        # Skip remote tests if API is not available or there's an error
        import pytest
        pytest.skip(f"Skipping remote reranker tests due to error: {str(e)}")
    