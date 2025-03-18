from pathlib import Path
from just_semantic_search.embeddings import EmbeddingModel, EmbeddingModelParams, load_sentence_transformer_from_enum, load_sentence_transformer_params_from_enum
from just_semantic_search.meta import IndexMultitonMeta, PydanticIndexMultitonMeta
from just_semantic_search.splitter_factory import create_splitter, SplitterType
from meilisearch_python_sdk.models.task import TaskInfo
from just_semantic_search.document import ArticleDocument, Document
from typing import List, Dict, Any, Literal, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
import numpy
import os

from meilisearch_python_sdk import AsyncClient, AsyncIndex, Client, Index
from meilisearch_python_sdk.errors import MeilisearchApiError
from meilisearch_python_sdk.index import SearchResults, Hybrid
from meilisearch_python_sdk.models.settings import MeilisearchSettings, UserProvidedEmbedder

import asyncio
import eliot
from eliot import start_action
import pydantic
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import wraps
import inspect
import time
import threading
from typing import ClassVar

# Define a retry decorator with exponential backoff using environment variables
def create_retry_decorator(func):
    return retry(
        stop=stop_after_attempt(int(os.getenv('RETRY_ATTEMPTS', 5))),
        wait=wait_exponential(
            multiplier=float(os.getenv('RETRY_MULTIPLIER', 1)),
            min=float(os.getenv('RETRY_MIN', 4)),
            max=float(os.getenv('RETRY_MAX', 10))
        ),
        before=lambda retry_state: eliot.log_message(
            message_type="retry_attempt",
            function=func.__name__,
            attempt=retry_state.attempt_number,
            error_type=str(retry_state.outcome.exception.__class__.__name__) if retry_state.outcome and retry_state.outcome.exception else "None"
        ) if retry_state.attempt_number > 1 else None
    )

def log_retry_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        retry_decorator = create_retry_decorator(func)
        
        if inspect.iscoroutinefunction(func):
            @retry_decorator
            async def async_wrapper(*args, **kwargs):
                with start_action(action_type="retry_action", function=func.__name__) as action:
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        action.log(
                            message_type="retry_failed",
                            error=str(e),
                            error_type=str(type(e).__name__)
                        )
                        raise
            return async_wrapper(*args, **kwargs)
        else:
            @retry_decorator
            def sync_wrapper(*args, **kwargs):
                with start_action(action_type="retry_action", function=func.__name__) as action:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        action.log(
                            message_type="retry_failed",
                            error=str(e),
                            error_type=str(type(e).__name__)
                        )
                        raise
            return sync_wrapper(*args, **kwargs)
    return wrapper


    


class MeiliBase(BaseModel):
    
    # Configuration fields
    host: str = Field(default=os.getenv("MEILISEARCH_HOST", "127.0.0.1")    , description="Meilisearch host address")
    port: int = Field(default=os.getenv("MEILISEARCH_PORT", 7700), description="Meilisearch port number")
    api_key: Optional[str] = Field(default=os.getenv("MEILISEARCH_API_KEY", "fancy_master_key"), description="Meilisearch API key for authentication")
    
    client: Optional[Client] = Field(default=None, exclude=True)
    client_async: Optional[AsyncClient] = Field(default=None, exclude=True)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Modify this field to use SkipValidation
    init_callback: Optional[Union[callable, pydantic.SkipValidation]] = Field(default=None, description="Optional callback function to run after initialization")

    @property
    def headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

  
    def model_post_init(self, __context) -> None:
        """Initialize clients and configure index after model initialization"""
        # Set model name
                # Add this at the end of model_post_init
        if self.init_callback is not None:
            self.init_callback(self)
        # Initialize clients
        base_url = f'http://{self.host}:{self.port}'
        with start_action(action_type="init_clients") as action:
            action.log(
                message_type="initializing_clients",
                base_url=base_url,
                api_key=self.api_key
            )
            self.client = Client(base_url, self.api_key)
            action.add_success_fields(
                message_type="clients_initialized",
                base_url=base_url,
                api_key=self.api_key
            )
        
    @log_retry_errors
    async def _configure_index(self):
        embedder = UserProvidedEmbedder(
            dimensions=1024,
            source="userProvided"
        )
        embedders = {
            self.model_name: embedder
        }
        settings = MeilisearchSettings(embedders=embedders, searchable_attributes=self.searchable_attributes)
        return await self.index_async.update_settings(settings)

    def all_indexes(self):
        return [key for key in self.client.get_all_stats().indexes.keys()]
    
    def non_empty_indexes(self):
        return [key for key, value in self.client.get_all_stats().indexes.items() if value.number_of_documents > 0]


    def delete_index(self):
        """
        synchronous version of delete_index_async
        """
        self.client.delete_index_if_exists(self.index_name)
    

    def get_url(self) -> str:
        return f'http://{self.host}:{self.port}'
    
    
        

# Module-level dictionary to store instances by index name
MEILIRAG_INSTANCES = {}
INSTANCE_LOCK = threading.RLock()  # Use RLock to allow reentrant locking

class MeiliAsyncRAG(MeiliBase):
    #WORK IN PROGRESS
    #it will be future fixure

    index_async: Optional[AsyncIndex] = Field(default=None, exclude=True)
    

    def get_loop(self):
        """Helper to get or create an event loop that works in all environments"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    

    def run_async(self, coro):
        """Helper method to run async code safely in all environments"""
        loop = self.get_loop()
        if loop.is_running():
            # Create a new loop for this operation if the current one is running
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
                asyncio.set_event_loop(loop)
        else:
            return loop.run_until_complete(coro)


    @log_retry_errors
    async def _init_index_async(self, 
                         create_index_if_not_exists: bool = True, 
                         recreate_index: bool = False) -> AsyncIndex:
        with start_action(action_type="init_index_async") as action:
            try:
                index = await self.client_async.get_index(self.index_name)
                if recreate_index:
                    action.log(
                        message_type="index_exists",
                        index_name=self.index_name,
                        recreate_index=True
                    )
                    deleted = await self.delete_index_async()
                    index = await self.client_async.create_index(self.index_name)
                    return index
                else:
                    action.add_success_fields(
                        message_type="index_exists",
                        index_name=self.index_name,
                        recreate_index=False
                    )
                    return index
            except MeilisearchApiError:
                if create_index_if_not_exists:
                    action.add_success_fields(
                        message_type="index_not_found",
                        index_name=self.index_name,
                        create_index_if_not_exists=True
                    )
                    index = await self.client_async.create_index(self.index_name)
                    await index.update_searchable_attributes(self.searchable_attributes)
                    await index.update_filterable_attributes(self.filterable_attributes)
                    return index
                else:
                    action.log(
                        message_type="index_not_found",
                        index_name=self.index_name,
                        create_index_if_not_exists=False
                    )
            return await self.client_async.get_index(self.index_name)


    @log_retry_errors
    async def add_documents_async(self, documents: List[ArticleDocument | Document], compress: bool = False) -> int:
        """Add ArticleDocument objects to the index."""
        with start_action(action_type="add documents") as action:
            documents_dict = [doc.model_dump(by_alias=True) for doc in documents]
            count = len(documents)
            result = await self.index_async.add_documents(documents_dict, primary_key=self.primary_key, compress=compress)
            action.add_success_fields(
                status=result.status,
                count = count
            )
            return result

    def model_post_init(self, __context) -> None:
        """Initialize clients and configure index after model initialization"""
        super().model_post_init(__context)
        self.client_async = AsyncClient(base_url=f'http://{self.host}:{self.port}', api_key=self.api_key)
        self.index_async = self.client_async.index(self.index_name)

        #self.index_async = self.run_async(
        #    self._init_index_async(self.create_index_if_not_exists, self.recreate_index)
        #)
        #self.run_async(self._configure_index())

    @log_retry_errors
    async def delete_index_async(self):
        return await self.client_async.delete_index_if_exists(self.index_name)
      
    @log_retry_errors
    async def _configure_async_index(self):
        embedder = UserProvidedEmbedder(
            dimensions=1024,
            source="userProvided"
        )
        embedders = {
            self.model_name: embedder
        }
        settings = MeilisearchSettings(embedders=embedders, searchable_attributes=self.searchable_attributes)
        return await self.index_async.update_settings(settings)

class MeiliRAG(MeiliBase):
    
    # RAG-specific fields
    index_name: str = Field(description="Name of the Meilisearch index")
    index: Optional[Index] = Field(default=None, exclude=True)
    model: EmbeddingModel = Field(default=EmbeddingModel.JINA_EMBEDDINGS_V3, description="Embedding model to use for vector search")
    embedding_model_params: EmbeddingModelParams = Field(default_factory=EmbeddingModelParams, description="Embedding model parameters")
    create_index_if_not_exists: bool = Field(default=os.getenv("MEILISEARCH_CREATE_INDEX_IF_NOT_EXISTS", True), description="Create index if it doesn't exist")
    recreate_index: bool = Field(default=os.getenv("MEILISEARCH_RECREATE_INDEX", False), description="Force recreate the index even if it exists")
    searchable_attributes: List[str] = Field(
        default=['title', 'abstract', 'text', 'content', 'source', "authors", "references"],
        description="List of attributes that can be searched"
    )
    filterable_attributes: List[str] = Field(
       default=['title', 'abstract', 'source', "authors", "references"],
        description="List of attributes that can be used for filtering"
    )

    # Primary key field for documents
    primary_key: str = Field(default="hash", description="Primary key field for documents")

    # Private fields for internal state
    model_name: Optional[str] = Field(default=None, exclude=True)
    st_model: Optional[SentenceTransformer] = Field(default=None, exclude=True)
    transformer_lock: ClassVar[threading.RLock] = threading.RLock()
  
    def model_post_init(self, __context) -> None:
        """Initialize clients and configure index after model initialization"""
        if self.init_callback is not None:
            self.init_callback(self)
        
        model_value = self.model.value
        self.embedding_model_params = load_sentence_transformer_params_from_enum(self.model)
        self.model_name = model_value.split("/")[-1].split("\\")[-1] if "/" in model_value or "\\" in model_value else model_value
        
        super().model_post_init(__context)
        # Instead of assigning, just call the method
        self.index = self._init_index(self.create_index_if_not_exists, self.recreate_index)
        self._configure_index()
    
    @property
    def sentence_transformer(self) -> SentenceTransformer:
        """Lazily load the sentence transformer model when it's first needed."""
        if self.st_model is None:
            with self.transformer_lock:
                # Check again to avoid race condition
                if self.st_model is None:
                    with start_action(action_type="lazy_load_sentence_transformer") as action:
                        action.log(
                            message_type="loading_sentence_transformer",
                            model=self.model.value
                        )
                        self.st_model = load_sentence_transformer_from_enum(self.model)
                        action.add_success_fields(
                            message_type="sentence_transformer_loaded",
                            model=self.model.value
                        )
        return self.st_model

    @classmethod
    def get_instance(cls, index_name: str, **kwargs):
        """Thread-safe method to get an existing MeiliRAG instance with reduced lock contention."""
        global MEILIRAG_INSTANCES, INSTANCE_LOCK
        
        # First check without lock
        if index_name in MEILIRAG_INSTANCES:
            return MEILIRAG_INSTANCES[index_name]
        
        # If not found, acquire lock and check again
        with INSTANCE_LOCK:
            if index_name in MEILIRAG_INSTANCES:
                return MEILIRAG_INSTANCES[index_name]
            
            # Create a new instance and store it in the pool
            instance = cls(index_name=index_name, **kwargs)
            MEILIRAG_INSTANCES[index_name] = instance
            return instance

    @log_retry_errors
    def _init_index(self, 
                         create_index_if_not_exists: bool = True, 
                         recreate_index: bool = False) -> Index:
        with start_action(action_type="init_index_sync") as action:
            try:
                index = self.client.get_index(self.index_name)
                if recreate_index:
                    action.log(
                        message_type="index_exists",
                        index_name=self.index_name,
                        recreate_index=True
                    )
                    self.client.delete_index_if_exists(self.index_name)
                    index = self.client.create_index(self.index_name)
                    return index
                else:
                    action.add_success_fields(
                        message_type="index_exists",
                        index_name=self.index_name,
                        recreate_index=False
                    )
                    return index
            except MeilisearchApiError:
                if create_index_if_not_exists:
                    action.add_success_fields(
                        message_type="index_not_found",
                        index_name=self.index_name,
                        create_index_if_not_exists=True
                    )
                    index = self.client.create_index(self.index_name)
                    index.update_searchable_attributes(self.searchable_attributes)
                    index.update_filterable_attributes(self.filterable_attributes)
                    return index
                else:
                    action.log(
                        message_type="index_not_found",
                        index_name=self.index_name,
                        create_index_if_not_exists=False
                    )
            return self.client.get_index(self.index_name)

    def add_documents(self, documents: List[ArticleDocument | Document], compress: bool = False):
        """Add documents synchronously by running the async method in the event loop."""
        with start_action(action_type="add documents") as action:
            documents_dict = [doc.model_dump(by_alias=True) for doc in documents]
            count = len(documents)
            result = self.index.add_documents(documents_dict, primary_key=self.primary_key, compress=compress)
            action.add_success_fields(
                status=result.status,
                count = count
            )
            return result
        
    
    def delete_by_source(self, source:str):
        """Delete documents by their sources from the MeiliRAG index."""
        self.index.delete_documents_by_filter(filters=f"source={source}")


    @log_retry_errors
    def get_documents(self, limit: int = 100, offset: int = 0):
        with start_action(action_type="get_documents") as action:
            result = self.index.get_documents(offset=offset, limit=limit)
            action.log(message_type="documents_retrieved", count=len(result.results))
            return result

        
    @log_retry_errors
    def search(self, 
            query: str | None = None,
            vector: Optional[Union[List[float], 'numpy.ndarray']] = None,
            semanticRatio: Optional[float] = os.getenv("MEILISEARCH_SEMANTIC_RATIO", 0.5),
            limit: int = os.getenv("MEILISEARCH_LIMIT", 100),
            offset: int = 0,
            filter: Any | None = None,
            facets: list[str] | None = None,
            attributes_to_retrieve: list[str] | None = None,
            attributes_to_crop: list[str] | None = None,
            crop_length: int = os.getenv("MEILISEARCH_CROP_LENGTH", 1000),
            attributes_to_highlight: list[str] | None = None,
            sort: list[str] | None = None,
            show_matches_position: bool = os.getenv("MEILISEARCH_SHOW_MATCHES_POSITION", False),
            highlight_pre_tag: str = "<em>",
            highlight_post_tag: str = "</em>",
            crop_marker: str = "...",
            matching_strategy: Literal["all", "last", "frequency"] = os.getenv("MEILISEARCH_MATCHING_STRATEGY", "last"),
            hits_per_page: int | None = None,
            page: int | None = None,
            attributes_to_search_on: list[str] | None = None,
            distinct: str | None = None,
            show_ranking_score: bool = os.getenv("MEILISEARCH_SHOW_RANKING_SCORE", True),
            show_ranking_score_details: bool = os.getenv("MEILISEARCH_SHOW_RANKING_SCORE_DETAILS", True),
            ranking_score_threshold: float | None = os.getenv("MEILISEARCH_RANKING_SCORE_THRESHOLD", None),
            locales: list[str] | None = None,
            sentence_transformer: Optional[SentenceTransformer] = None, 
            **kwargs
        ) -> SearchResults:
        """Search for documents in the index.
        
        Args:
            query (Optional[str]): Search query text
            vector (Optional[Union[List[float], numpy.ndarray]]): Vector embedding for semantic search
            limit (Optional[int]): Maximum number of results to return
            retrieve_vectors (Optional[bool]): Whether to return vector embeddings
            semanticRatio (Optional[float]): Ratio between semantic and keyword search
            show_ranking_score (Optional[bool]): Show ranking scores in results
            show_matches_position (Optional[bool]): Show match positions in results
            
        Returns:
            SearchResults: Search results including hits and metadata
        """
        
        # Convert numpy array to list if necessary
        if vector is not None and hasattr(vector, 'tolist'):
            vector = vector.tolist()
        
        # First check if semanticRatio is 0.0 - no need for vectorization
        if semanticRatio <= 0.0:
            with start_action(action_type="execute_search_query_text_only") as action:
                action.log(message_type="search_query_start", 
                          query_text=query, 
                          limit=limit,
                          semantic_ratio=semanticRatio)
                results = self.index.search(query,
                    offset=offset,
                    limit=limit,
                    filter=filter,
                    facets=facets,
                    attributes_to_retrieve=attributes_to_retrieve,
                    attributes_to_crop=attributes_to_crop,
                    crop_length=crop_length,
                    attributes_to_highlight=attributes_to_highlight,
                    sort=sort,
                    show_matches_position=show_matches_position,
                    highlight_pre_tag=highlight_pre_tag,
                    highlight_post_tag=highlight_post_tag,
                    crop_marker=crop_marker,
                    matching_strategy=matching_strategy,
                    hits_per_page=hits_per_page,
                    page=page,
                    attributes_to_search_on=attributes_to_search_on,
                    distinct=distinct,
                    show_ranking_score=show_ranking_score,
                    show_ranking_score_details=show_ranking_score_details,
                    ranking_score_threshold=ranking_score_threshold,
                    locales=locales)
                return results
        
        # Only initialize sentence_transformer and generate vectors if semanticRatio > 0
        if vector is None:
            if sentence_transformer is None:
                sentence_transformer = self.st_model if self.st_model is not None else self.sentence_transformer
            
            kwargs.update(self.embedding_model_params.retrival_query)
            with start_action(action_type="encode_query") as action:
                # Check if CUDA is available and being used
                try:
                    import torch
                    device = next(sentence_transformer.parameters()).device
                    is_cuda = device.type == 'cuda'
                    cuda_device_name = torch.cuda.get_device_name(device) if is_cuda else "N/A"
                except Exception as e:
                    # Handle the error gracefully
                    action.log(
                        message_type="cuda_detection_error",
                        error_type=str(type(e).__name__),
                        error=str(e)
                    )
                    is_cuda = False
                    cuda_device_name = "N/A"
                
                action.log(
                    message_type="encoding_query_start", 
                    query_length=len(query) if query else 0,
                    device_type=device.type,
                    is_cuda=is_cuda,
                    cuda_device=cuda_device_name if is_cuda else None
                )
                
                start_time = time.time()
                vector = sentence_transformer.encode(query, **kwargs).tolist()
                encoding_time = time.time() - start_time
                # Format time as minutes:seconds
                minutes = int(encoding_time // 60)
                seconds = encoding_time % 60
                time_formatted = f"{minutes}:{seconds:.2f}"
                action.add_success_fields(
                    message_type="encoding_query_complete",
                    encoding_time=time_formatted,
                    encoding_time_seconds=encoding_time,
                    vector_dimensions=len(vector) if vector else 0,
                    device_type=device.type
                )
        
        hybrid = Hybrid(
            embedder=self.model_name,
            semanticRatio=semanticRatio
        )
        
        with start_action(action_type="execute_search_query") as action:
            action.log(message_type="search_query_start", 
                      query_text=query, 
                      limit=limit,
                      semantic_ratio=semanticRatio)
            search_start_time = time.time()
            
            results = self.index.search(
                query,
                offset=offset,
                limit=limit,
                filter=filter,
                facets=facets,
                attributes_to_retrieve=attributes_to_retrieve,
                attributes_to_crop=attributes_to_crop,
                crop_length=crop_length,
                attributes_to_highlight=attributes_to_highlight,
                sort=sort,
                show_matches_position=show_matches_position,
                highlight_pre_tag=highlight_pre_tag,
                highlight_post_tag=highlight_post_tag,
                crop_marker=crop_marker,
                matching_strategy=matching_strategy,
                hits_per_page=hits_per_page,
                page=page,
                attributes_to_search_on=attributes_to_search_on,
                distinct=distinct,
                show_ranking_score=show_ranking_score,
                show_ranking_score_details=show_ranking_score_details,
                ranking_score_threshold=ranking_score_threshold,
                vector=vector,
                hybrid=hybrid,
                locales=locales
            )
            
            search_time = time.time() - search_start_time
            # Format time as minutes:seconds
            minutes = int(search_time // 60)
            seconds = search_time % 60
            search_time_formatted = f"{minutes}:{seconds:.2f}"
            
            action.add_success_fields(
                message_type="search_query_complete",
                search_time=search_time_formatted,
                search_time_seconds=search_time,
                hits_count=len(results.hits) if hasattr(results, 'hits') else 0
            )
            
            return results

    @log_retry_errors
    def _configure_index(self):
        embedder = UserProvidedEmbedder(
            dimensions=1024,
            source="userProvided"
        )
        embedders = {
            self.model_name: embedder
        }
        settings = MeilisearchSettings(embedders=embedders, searchable_attributes=self.searchable_attributes)
        return self.index.update_settings(settings)
        
    def index_folder(
        self,
        folder: Path,
        splitter: SplitterType = SplitterType.TEXT
    ) -> None:
        """Index documents from a folder using the provided MeiliRAG instance."""
        with start_action(message_type="index_folder", folder=str(folder)) as action:
            sentence_transformer_model = load_sentence_transformer_from_enum(self.model)
            splitter_instance = create_splitter(splitter, sentence_transformer_model)
            documents = splitter_instance.split_folder(folder)
            result = self.add_documents(documents)
            action.add_success_fields(
                message_type="index_folder_complete",
                index_name=self.index_name,
                documents_added_count=len(documents)
            )
            return result


    