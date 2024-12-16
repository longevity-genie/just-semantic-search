import typer
import meilisearch
import os
from dotenv import load_dotenv
from just_semantic_search.meili.rag import *
import requests
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


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

class SearchResult(BaseModel):
    hits: List[Dict[str, Any]]
    processing_time_ms: int = Field(alias="processingTimeMs")
    estimated_total_hits: int = Field(alias="estimatedTotalHits")
    query: str
    model_config = ConfigDict(extra='allow')

class MeiliRAG:
    def __init__(
        self, 
        config: MeiliConfig,
        model_name: str
    ):
        self.config = config
        self.client = meilisearch.Client(config.get_url(), config.api_key)
        self.model_name = model_name
        
        # Enable vector store during initialization
        if not self._enable_vector_store():
            typer.echo("Warning: Failed to enable vector store feature during initialization")

    def _enable_vector_store(self) -> bool:
        """Enable vector store feature in Meilisearch."""
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

    def add_documents(self, index_name: str, documents: List[Dict[str, Any]]) -> int:
        try:
            index = self.client.get_index(index_name)
        except Exception as e:
            print(f"Index not found, creating new index '{index_name}'...")
            index = self.client.create_index(index_name, {'primaryKey': 'id'})
        
        index.add_documents(documents)
        return len(documents)

    def search(self, index_name: str, query: str) -> SearchResult:
        index = self.client.index(index_name)
        results = index.search(query)
        return SearchResult.model_validate(results)

    def delete_index(self, index_name: str) -> None:
        self.client.delete_index(index_name)

    def configure_embedder(
        self,
        index_name: str,
        source: str = "userProvided",
        dimensions: Optional[int] = 1024
    ) -> bool:
        """Configure an embedder in Meilisearch with specified name."""
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

    def create_index(
        self, 
        index_name: str, 
        primary_key: str = "id"
    ) -> None:
        # Remove the vector store check since it's done in initialization
        index = self.client.create_index(index_name, {'primaryKey': primary_key})
        self.configure_embedder(
            index_name=index_name, 
            model_name=self.model_name,
            name=self.model_name
        )
        return index