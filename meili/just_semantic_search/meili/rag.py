from just_semantic_search.embeddings import EmbeddingModel, EmbeddingModelParams, load_sentence_transformer_params_from_enum
from just_semantic_search.meili.utils.retry import create_retry_decorator
from meilisearch_python_sdk.models.task import TaskInfo
from just_semantic_search.document import ArticleDocument, Document
from just_semantic_search.meili.rag import *
from typing import List, Dict, Any, Literal, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
import numpy
import os

from meilisearch_python_sdk import AsyncClient, AsyncIndex
from meilisearch_python_sdk import Client
from meilisearch_python_sdk.errors import MeilisearchApiError
from meilisearch_python_sdk.index import SearchResults, Hybrid
from meilisearch_python_sdk.models.settings import MeilisearchSettings, UserProvidedEmbedder

import asyncio
from eliot import start_action, log_message
from sentence_transformers import SentenceTransformer

# Define a retry decorator with exponential backoff using environment variables
retry_decorator = create_retry_decorator(
    attempts=int(os.getenv('RETRY_ATTEMPTS', 5)),
    multiplier=float(os.getenv('RETRY_MULTIPLIER', 1)),
    min_wait=float(os.getenv('RETRY_MIN', 4)),
    max_wait=float(os.getenv('RETRY_MAX', 10))
)

class MeiliRAG(BaseModel):
    # Configuration fields
    host: str = Field(default="127.0.0.1", description="Meilisearch host address")
    port: int = Field(default=7700, description="Meilisearch port number")
    api_key: Optional[str] = Field(default="fancy_master_key", description="Meilisearch API key for authentication")
    
    # RAG-specific fields
    index_name: str = Field(description="Name of the Meilisearch index")
    model: EmbeddingModel = Field(description="Embedding model to use for vector search")
    embedding_model_params: EmbeddingModelParams = Field(default_factory=EmbeddingModelParams, description="Embedding model parameters")
    create_index_if_not_exists: bool = Field(default=True, description="Create index if it doesn't exist")
    recreate_index: bool = Field(default=False, description="Force recreate the index even if it exists")
    searchable_attributes: List[str] = Field(
        default=['title', 'abstract', 'text', 'content', 'source'],
        description="List of attributes that can be searched"
    )
    primary_key: str = Field(default="hash", description="Primary key field for documents")

    # Add this new field near the other configuration fields
    init_callback: Optional[callable] = Field(default=None, description="Optional callback function to run after initialization")

    # Private fields for internal state
    model_name: Optional[str] = Field(default=None, exclude=True)
    client: Optional[Client] = Field(default=None, exclude=True)
    client_async: Optional[AsyncClient] = Field(default=None, exclude=True)
    index_async: Optional[AsyncIndex] = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context) -> None:
        """Initialize clients and configure index after model initialization"""
        # Set model name
                # Add this at the end of model_post_init
        if self.init_callback is not None:
            self.init_callback(self)
        model_value = self.model.value
        self.embedding_model_params = load_sentence_transformer_params_from_enum(self.model)
        self.model_name = model_value.split("/")[-1].split("\\")[-1] if "/" in model_value or "\\" in model_value else model_value
        
        # Initialize clients
        base_url = f'http://{self.host}:{self.port}'
        self.client = Client(base_url, self.api_key)
        self.client_async = AsyncClient(base_url, self.api_key)
        
        self.index_async = self.run_async(
            self._init_index_async(self.create_index_if_not_exists, self.recreate_index)
        )
        self.run_async(self._configure_index())
        

    @property
    def headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get_loop(self):
        """Helper to get or create an event loop that works with both CLI and Jupyter"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def run_async(self, coro):
        """Helper method to run async code in both Jupyter and non-Jupyter environments"""
        loop = self.get_loop()
        return loop.run_until_complete(coro)

    @retry_decorator
    async def delete_index_async(self):
        return await self.client_async.delete_index_if_exists(self.index_name)


    def delete_index(self):
        """
        synchronous version of delete_index_async
        """
        return self.get_loop().run_until_complete(self.delete_index_async())
    

    @retry_decorator
    async def _init_index_async(self, 
                         create_index_if_not_exists: bool = True, 
                         recreate_index: bool = False) -> AsyncIndex:
        with start_action(action_type="init_index_async") as action:
            try:
                index = await self.client_async.get_index(self.index_name)
                if recreate_index:
                    log_message(
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
                    return index
                else:
                    action.log(
                        message_type="index_not_found",
                        index_name=self.index_name,
                        create_index_if_not_exists=False
                    )
            return await self.client_async.get_index(self.index_name)

    def get_url(self) -> str:
        return f'http://{self.host}:{self.port}'

        
    @retry_decorator
    async def add_documents_async(self, documents: List[ArticleDocument | Document], compress: bool = False) -> int:
        """Add ArticleDocument objects to the index."""
        with start_action(action_type="add documents") as action:
            documents_dict = [doc.model_dump(by_alias=True) for doc in documents]
            count = len(documents)
            result =  await self.add_document_dicts_async(documents_dict, compress=compress)
            #self.client.index(self.index_name).get_update_status(result.task_uid)
            action.add_success_fields(
                status=result.status,
                count = count
            )
            return result
            
    def add_documents(self, documents: List[ArticleDocument | Document], compress: bool = False):
        """Add documents synchronously by running the async method in the event loop."""
        result = self.run_async(
            self.add_documents_async(documents, compress=compress)
        )
        return result


    @retry_decorator
    def get_documents(self, limit: int = 100, offset: int = 0):
        with start_action(action_type="get_documents") as action:
            result = self.index.get_documents(offset=offset, limit=limit)
            action.log(message_type="documents_retrieved", count=len(result.results))
            return result

    @retry_decorator
    async def add_document_dicts_async(self, documents: List[Dict[str, Any]], compress: bool = False) -> TaskInfo:
        with start_action(action_type="add_document_dicts_async") as action:
            test = documents[0]
            result = await self.index_async.add_documents(documents, primary_key=self.primary_key, compress=compress)
            return result


    @retry_decorator
    def search(self, 
            query: str | None = None,
            vector: Optional[Union[List[float], 'numpy.ndarray']] = None,
            semanticRatio: Optional[float] = 0.5,
            limit: int = 100,
            offset: int = 0,
            filter: Any | None = None,
            facets: list[str] | None = None,
            attributes_to_retrieve: list[str] | None = None,
            attributes_to_crop: list[str] | None = None,
            crop_length: int = 1000,
            attributes_to_highlight: list[str] | None = None,
            sort: list[str] | None = None,
            show_matches_position: bool = False,
            highlight_pre_tag: str = "<em>",
            highlight_post_tag: str = "</em>",
            crop_marker: str = "...",
            matching_strategy: Literal["all", "last", "frequency"] = "last",
            hits_per_page: int | None = None,
            page: int | None = None,
            attributes_to_search_on: list[str] | None = None,
            distinct: str | None = None,
            show_ranking_score: bool = True,
            show_ranking_score_details: bool = True,
            ranking_score_threshold: float | None = None,
            locales: list[str] | None = None,
            model: Optional[SentenceTransformer] = None, 
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
        else:
            if model is not None:
                kwargs.update(self.embedding_model_params.retrival_query)
                vector = model.encode(query, **kwargs).tolist()
        
        hybrid = Hybrid(
            embedder=self.model_name,
            semanticRatio=semanticRatio
        )
        
        return self.index.search(
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

    @retry_decorator
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


    @property
    @retry_decorator
    def index(self):
        """Get the Meilisearch index.
        
        Returns:
            Index: Meilisearch index object
            
        Raises:
            ValueError: If index not found
        """
        try:
            return self.client.get_index(self.index_name)
        except MeilisearchApiError as e:
            raise ValueError(f"Index '{self.index_name}' not found: {e}")
    
