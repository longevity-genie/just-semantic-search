from just_semantic_search.embeddings import EmbeddingModel
from meilisearch_python_sdk.models.task import TaskInfo
from just_semantic_search.document import ArticleDocument, Document
import os
from dotenv import load_dotenv
from just_semantic_search.meili.rag import *
import requests
from typing import List, Dict, Any, Literal, Mapping, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
import numpy

from meilisearch_python_sdk import AsyncClient, AsyncIndex
from meilisearch_python_sdk import Client
from meilisearch_python_sdk.errors import MeilisearchApiError
from meilisearch_python_sdk.index import SearchResults, Hybrid
from meilisearch_python_sdk.models.settings import MeilisearchSettings, UserProvidedEmbedder

import asyncio
from eliot import start_action, log_message
from sentence_transformers import SentenceTransformer


class MeiliConfig(BaseModel):
    host: str = Field(default="127.0.0.1", description="Meilisearch host")
    port: int = Field(default=7700, description="Meilisearch port")
    api_key: Optional[str] = Field(default="fancy_master_key", description="Meilisearch API key")
    
    def get_url(self) -> str:
        return f'http://{self.host}:{self.port}'
    
    @property
    def headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    

class MeiliRAG:
    
    def get_loop(self):
        """Helper to get or create an event loop"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    
    
    def __init__(
        self,
        index_name: str, 
        model: EmbeddingModel,
        config: MeiliConfig,
        create_index_if_not_exists: bool = True,
        recreate_index: bool = False, 
        searchable_attributes: List[str] = ['title', 'abstract', 'text', 'content', 'source'],
        primary_key: str = "hash"
    ):
        with start_action(action_type="init_rag") as action:
            """Initialize MeiliRAG instance.
            
            Args:
                index_name (str): Name of the Meilisearch index
                model_name (str): Name of the embedding model
                config (MeiliConfig): Meilisearch configuration
                create_index_if_not_exists (bool): Create index if it doesn't exist
                recreate_index (bool): Force recreate the index even if it exists
            """
            self.config = config
            #self.client = meilisearch.Client(config.get_url(), config.api_key)
            
            self.client_async  = AsyncClient(config.get_url(), config.api_key)
            self.client = Client(config.get_url(), config.api_key)
            
            model_value = model.value
            if "/" in model_value or "\\" in model_value:
                self.model_name = model_value.split("/")[-1].split("\\")[-1]
            else:
                self.model_name = model_value
            self.index_name = index_name
            self.primary_key = primary_key
            self.searchable_attributes = searchable_attributes
     
            if not self._enable_vector_store():
                action.log(message_type="warning", message="Failed to enable vector store feature during initialization")
            self.index_async = self.get_loop().run_until_complete(
                self._init_index_async(create_index_if_not_exists, recreate_index)
            )
            self.get_loop().run_until_complete(self._configure_index())

    async def delete_index_async(self):
        return await self.client_async.delete_index_if_exists(self.index_name)


    def delete_index(self):
        """
        synchronous version of delete_index_async
        """
        return self.get_loop().run_until_complete(self.delete_index_async())
    

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



    def _enable_vector_store(self) -> bool:
        """Enable vector store feature in Meilisearch."""
        response = requests.patch(
                f'{self.config.get_url()}/experimental-features',
                json={'vectorStore': True, 'metrics': True},
                headers=self.config.headers,
                verify=True
            )
        
        response.raise_for_status()
        return True
        return False
        
    async def add_documents_async(self, documents: List[ArticleDocument | Document], compress: bool = False) -> int:
        """Add ArticleDocument objects to the index."""
        with start_action(action_type="add documents") as action:
            documents_dict = [doc.model_dump(by_alias=True) for doc in documents]
            count = len(documents)
            result =  await self.add_document_dicts_async(documents_dict, compress=compress)
            action.add_success_fields(
                status=result.status,
                count = count
            )
            return result
            
    
    def add_documents(self, documents: List[ArticleDocument | Document], compress: bool = False):
        """Add documents synchronously by running the async method in the event loop.
        
        Args:
            documents (List[ArticleDocument | Document]): List of documents to add
            compress (bool): Whether to compress the documents
            
        Returns:
            The result from add_documents_async
        """
        result = self.get_loop().run_until_complete(
            self.add_documents_async(documents, compress=compress)
        )
        return result


    def get_documents(self, limit: int = 100, offset: int = 0):
        with start_action(action_type="get_documents") as action:
            result = self.index.get_documents(offset=offset, limit=limit)
            action.log(message_type="documents_retrieved", count=len(result.results))
            return result

    async def add_document_dicts_async(self, documents: List[Dict[str, Any]], compress: bool = False) -> TaskInfo:
        result = await self.index_async.add_documents(documents, primary_key=self.primary_key, compress=compress)
        return result


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
            model: Optional[SentenceTransformer] = None
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
                vector = model.encode(query).tolist()
        
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
