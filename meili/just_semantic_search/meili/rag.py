from just_semantic_search.document import ArticleDocument, Document
import typer
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
        model_name: str,
        config: MeiliConfig,
        create_index_if_not_exists: bool = True,
        recreate_index: bool = False, 
        searchable_attributes: List[str] = ['title', 'abstract', 'text', 'content', 'source'],
        primary_key: str = "hash"
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
        self.primary_key = primary_key
        self.searchable_attributes = searchable_attributes
        if not self._enable_vector_store():
            typer.echo("Warning: Failed to enable vector store feature during initialization")
        self.index_async = self.get_loop().run_until_complete(
            self.init_index(create_index_if_not_exists, recreate_index)
        )
        self.get_loop().run_until_complete(self.configure_index())


    async def init_index(self, create_index_if_not_exists: bool = True, recreate_index: bool = False) -> AsyncIndex:
        try:
            index = await self.client_async.get_index(self.index_name)
            if recreate_index:
                typer.echo(f"Index '{self.index_name}' already exists, because recreate_index=True we will delete it and create a new one")
                deleted = await self.client_async.delete_index_if_exists(self.index_name)
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
            typer.echo(f"An error occurred while enabling vector store: {e}")
            return False
        
    async def add_documents_async(self, documents: List[ArticleDocument | Document], compress: bool = False) -> int:
        """Add ArticleDocument objects to the index.
        
        Args:
            documents (List[ArticleDocument | Document]): List of documents to add
            compress (bool): Whether to compress the documents
            
        Returns:
            int: Number of documents added
            
        Raises:
            MeiliSearchApiError: If documents cannot be added
        """
        try:
            documents_dict = [doc.model_dump(by_alias=True) for doc in documents]
            return await self.add_document_dicts_async(documents_dict, compress=compress)
        except Exception as e:
            typer.echo(f"Error converting documents to dictionaries: {str(e)}")
            raise

    def add_documents_dicts_sync(self, documents: List[Dict[str, Any]], compress: bool = False):
        return self.client.index(self.index_name).add_documents(documents, primary_key=self.primary_key, compress=compress)
    
    def add_documents_sync(self, documents: List[ArticleDocument | Document], compress: bool = False):
        docs = [doc.model_dump(by_alias=True) for doc in documents]
        return self.add_documents_dicts_sync(docs, compress=compress)

    def get_documents(self):
        return self.index.get_documents()

    async def add_document_dicts_async(self, documents: List[Dict[str, Any]], compress: bool = False):
        """Add dictionary documents to the index.
        
        Args:
            documents (List[Dict[str, Any]]): List of document dictionaries
            compress (bool): Whether to compress the documents
            
        Returns:
            int: Number of documents added
            
        Raises:
            MeiliSearchApiError: If documents cannot be added
        """
        try:
            typer.echo(f"Attempting to add {len(documents)} documents to index '{self.index_name}'")
            result = await self.index_async.add_documents(documents, primary_key=self.primary_key, compress=compress)
            typer.echo(f"Successfully added documents. Result: {result}")
            return result
        except MeilisearchApiError as e:
            typer.echo(f"Meilisearch API Error while adding documents:")
            typer.echo(f"Error type: {type(e).__name__}")
            typer.echo(f"Error message: {str(e)}")
            typer.echo(f"Error code: {getattr(e, 'code', 'unknown')}")
            raise
        except Exception as e:
            typer.echo(f"Unexpected error while adding documents:")
            typer.echo(f"Error type: {type(e).__name__}")
            typer.echo(f"Error message: {str(e)}")
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

    async def configure_index(self):
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
