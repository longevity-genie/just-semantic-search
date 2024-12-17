from just_semantic_search.document import ArticleDocument, Document
import typer
import meilisearch
import os
from dotenv import load_dotenv
from just_semantic_search.meili.rag import *
import requests
from typing import List, Dict, Any, Literal, Mapping, Optional, Union
import meilisearch
from pydantic import BaseModel, Field, ConfigDict
import numpy

from meilisearch_python_sdk import AsyncClient, AsyncIndex
from meilisearch_python_sdk import Client
from meilisearch_python_sdk.index import SearchResults, Hybrid

import asyncio


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
    
"""
class SearchResult(BaseModel):
    hits: List[Dict[str, Any]]
    processing_time_ms: int = Field(alias="processingTimeMs")
    estimated_total_hits: int = Field(alias="estimatedTotalHits")
    query: str
    model_config = ConfigDict(extra='allow')
"""     
    

class MeiliRAG:
    
    def __init__(
        self,
        index_name: str, 
        model_name: str,
        config: MeiliConfig,
        create_index_if_not_exists: bool = True,
        recreate_index: bool = False, 
        indexable_attributes: List[str] = ['title', 'abstract', 'text', 'content', 'source']
    ):
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
        
        self.model_name = model_name
        self.index_name = index_name
        self.indexable_attributes = indexable_attributes
        self.index_async: AsyncIndex = asyncio.run(self.init_index(create_index_if_not_exists, recreate_index))
        


    async def init_index(self, create_index_if_not_exists: bool = True, recreate_index: bool = False) -> AsyncIndex:
        if not self._enable_vector_store():
            typer.echo("Warning: Failed to enable vector store feature during initialization")
        try:
            index = await self.client_async.get_index(self.index_name)
            if recreate_index:
                typer.echo(f"Index '{self.index_name}' already exists, because recreate_index=True we will delete it and create a new one")
                deleted = self.client.delete_index_if_exists(self.index_name)
                index = await self.client_async.create_index(self.index_name)
                return index
            else:
                typer.echo(f"Index '{self.index_name}' already exists, skipping creation")
                return index
        except MeilisearchApiError:
            if create_index_if_not_exists:
                typer.echo(f"Index '{self.index_name}' not found, creating...")
                index = await self.client_async.create_index(self.index_name)
                await index.update_searchable_attributes(self.indexable_attributes)
                return index
            else:
                typer.echo(f"Index '{self.index_name}' not found and create_index_if_not_exists=False")
        return await self.client_async.get_index(self.index_name)



    def _enable_vector_store(self) -> bool:
        """Enable vector store feature in Meilisearch.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = requests.patch(
                f'{self.config.get_url()}/experimental-features',
                json={'vectorStore': True, 'metrics': True},
                headers=self.config.headers,
                verify=True
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            return False
        
    async def add_documents(self, documents: List[ArticleDocument | Document]) -> int:
        """Add ArticleDocument objects to the index.
        
        Args:
            documents (List[ArticleDocument | Document]): List of documents to add
            
        Returns:
            int: Number of documents added
        """
        documents_dict = [doc.model_dump(by_alias=True) for doc in documents]
        return await self.add_document_dicts(documents_dict)
    
    def add_documents_sync(self, documents: List[ArticleDocument | Document]) -> int:
        return asyncio.run(self.add_document_dicts(documents))

    async def add_document_dicts(self, documents: List[Dict[str, Any]]) -> int:
        """Add dictionary documents to the index.
        
        Args:
            documents (List[Dict[str, Any]]): List of document dictionaries
            
        Returns:
            int: Number of documents added
            
        Raises:
            MeiliSearchApiError: If documents cannot be added
        """
        try:
            return await self.index_async.add_documents(documents)
        except MeilisearchApiError as e:
            print(f"Error adding documents: {e}")
            raise



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

    def configure_embedder(
        self,
        index_name: str,
        source: str = "userProvided",
        dimensions: Optional[int] = 1024
    ) -> bool:
        """Configure an embedder in Meilisearch.
        
        Args:
            index_name (str): Name of the index to configure
            source (str): Source of embeddings ('userProvided' or model URL)
            dimensions (Optional[int]): Dimensions of embedding vectors
            
        Returns:
            bool: True if configuration successful, False otherwise
        """
        embedder_config = {
            'source': source,
        }
        
        # Add appropriate fields based on source
        if source == "userProvided":
            embedder_config.update({
                'dimensions': dimensions
            })
        else:
            embedder_config.update({
                'modelUrl': self.model_name,
                'documentTemplate': "{text}"
            })

        js = {
            "embedders": {
                self.model_name: embedder_config
            }
        }
        
        try:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.patch(
                f'{self.config.get_url()}/indexes/{index_name}/settings',
                json=js,
                headers=headers,
                verify=True
            )
            response.raise_for_status()
            typer.echo(f"Successfully configured embedder '{self.model_name}'")
            return True
        except requests.exceptions.RequestException as e:
            typer.echo(f"An error occurred while configuring embedder: {e} \n JSON: {js}")
            return False


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
