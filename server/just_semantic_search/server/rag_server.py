import os
from typing import List, Dict, Optional, Union
from fastapi import Query
from just_semantic_search.embeddings import EmbeddingModel
from just_semantic_search.meili.rag import MeiliRAG
from just_semantic_search.meili.tools import search_documents, all_indexes
from just_semantic_search.server import indexing
from pydantic import BaseModel, Field
from just_agents.base_agent import BaseAgent
from just_agents.web.rest_api import AgentRestAPI
from eliot import start_task
from just_semantic_search.server.rag_agent import DEFAULT_RAG_AGENT
from pathlib import Path
import uvicorn
from just_agents.web.config import WebAgentConfig
import typer
from pycomfort.logging import to_nice_stdout
from just_agents import llm_options
from just_semantic_search.document import Document
from just_semantic_search.server.indexing import index_md_txt

class RAGServerConfig(WebAgentConfig):
    """Configuration for the RAG server"""
    port: int = Field(
        default_factory=lambda: int(os.getenv("APP_PORT", "8090").split()[0]),
        description="Port number for the server",
        ge=1024,  # Recommended to use ports above 1024 for non-root users
        le=65535,
        examples=[8088, 8000, 5000]
    )
    
    host: str = Field(
        default_factory=lambda: os.getenv("APP_HOST", "0.0.0.0").split()[0],
        description="Host address to bind the server to",
        examples=["0.0.0.0", "127.0.0.1"]
    )
   
    

env_config = RAGServerConfig()
app = typer.Typer()

class SearchRequest(BaseModel):
    """Request model for basic semantic search"""
    query: str = Field(example="Glucose predictions models for CGM")
    index: str = Field(example="glucosedao")
    limit: int = Field(default=10, ge=1, example=30)
    semantic_ratio: float = Field(default=0.5, ge=0.0, le=1.0, example=0.5)
   

class SearchAgentRequest(BaseModel):
    """Request model for RAG-based advanced search"""
    query: str = Field(example="Glucose predictions models for CGM")
    index: Optional[str] = Field(default=None, example="glucosedao")
    additional_instructions: Optional[str] = Field(default=None, example="You must always provide quotes from evidence followed by the sources (not in the end but immediately after the quote)")


class RAGServer(AgentRestAPI):
    """Extended REST API implementation that adds RAG (Retrieval-Augmented Generation) capabilities"""

    def __init__(self, 
                 agents: Optional[Dict[str, BaseAgent]] = None,
                 agent_config: Optional[Path | str] = None,
                 agent_section: Optional[str] = None,
                 agent_parent_section: Optional[str] = None,
                 debug: bool = False,
                 title: str = "Just-Agent endpoint",
                 description: str = "OpenAI-compatible API endpoint for Just-Agents",
                 *args, **kwargs):
        if agents is not None:
            kwargs["agents"] = agents
        super().__init__(
            agent_config=agent_config,
            agent_section=agent_section,
            agent_parent_section=agent_parent_section,
            debug=debug,
            title=title,
            description=description,
            *args, **kwargs
        )
        self._indexes = None
        self._configure_rag_routes()

    @property
    def indexes(self) -> List[str]:
        """Lazy property that returns cached list of indexes or fetches them if not cached"""
        if self._indexes is None:
            self._indexes = self.list_indexes()
        return self._indexes

    def _configure_rag_routes(self):
        """Configure RAG-specific routes"""
        self.post("/search", description="Perform semantic search")(self.search)
        self.post("/search_agent", description="Perform advanced RAG-based search")(self.search_agent)
        self.post("/list_indexes", description="Get all indexes")(self.list_indexes)
        self.post("/index_markdown_folder", description="Index a folder with markdown files")(self.index_markdown_folder)

    def search(self, request: SearchRequest) -> list[str]:
        """
        Perform a semantic search.
        
        Args:
            request: SearchRequest object containing search parameters
            
        Returns:
            List of matching documents with their metadata
        """
        with start_task(action_type="rag_server_search", 
                       query=request.query, 
                       index=request.index, 
                       limit=request.limit) as action:
            action.log("performing search")
            return search_documents(
                query=request.query,
                index=request.index,
                limit=request.limit,
                semantic_ratio=request.semantic_ratio
            )

    def search_agent(self, request: SearchAgentRequest) -> str:
        """
        Perform an advanced search using the RAG agent that can provide contextual answers.
        
        Args:
            request: SearchAgentRequest object containing the query, optional index, and additional instructions
            
        Returns:
            A detailed response from the RAG agent incorporating retrieved documents
        """
        with start_task(action_type="rag_server_advanced_search", query=request.query) as action:
            action.log("performing advanced RAG search")
            indexes = self.indexes if request.index is None else [request.index]
            query = f"Search the following query:```\n{request.query}\n```\nYou can only search in the following indexes: {indexes}"
            if request.additional_instructions is not None:
                query += f"\nADDITIONAL INSTRUCTIONS: {request.additional_instructions}"
            result = DEFAULT_RAG_AGENT.query(query)
            return result
    
    def list_indexes(self, non_empty: bool = True) -> List[str]:
        """
        Get all indexes and update the cache.
        """
        self._indexes = all_indexes(non_empty=non_empty)
        return self._indexes
    
    def index_markdown_folder(self, folder: str, index_name: str) -> str:
        """
        Indexes a folder with markdown files. The server should have access to the folder.
        Uses defensive checks for documents that might be either dicts or Document instances.
        Reports errors to Eliot logs without breaking execution; problematic documents are skipped.
        """
        
        with start_task(action_type="rag_server_index_markdown_folder", folder=folder, index_name=index_name) as action:
            folder_path = Path(folder)
            if not folder_path.exists():
                msg = f"Folder {folder} does not exist or the server does not have access to it"
                action.log(msg)
                return msg
            
            model_str = os.getenv("EMBEDDING_MODEL", EmbeddingModel.JINA_EMBEDDINGS_V3.value)
            model = EmbeddingModel(model_str)

            max_seq_length: Optional[int] = os.getenv("INDEX_MAX_SEQ_LENGTH", 3600)
            characters_for_abstract: int = os.getenv("INDEX_CHARACTERS_FOR_ABSTRACT", 10000)
            
            # Create and return RAG instance with conditional recreate_index
            # It should use default environment variables for host, port, api_key, create_index_if_not_exists, recreate_index
            rag = MeiliRAG(
                index_name=index_name,
                model=model,        # The embedding model used for the search
            )
            options = llm_options.GEMINI_2_FLASH if self.agents is None else list(self.agents.values())[0].llm_options
            docs = index_md_txt(rag, folder, max_seq_length, characters_for_abstract, options=options)
            sources = []
            valid_docs_count = 0
            error_count = 0

            for doc in docs:
                try:
                    if isinstance(doc, dict):
                        source = doc.get("source")
                        if source is None:
                            raise ValueError(f"Document (dict) missing 'source' key: {doc}")
                    elif isinstance(doc, Document):
                        source = getattr(doc, "source", None)
                        if source is None:
                            raise ValueError(f"Document instance missing 'source' attribute: {doc}")
                    else:
                        raise TypeError(f"Unexpected document type: {type(doc)} encountered in documents list")

                    sources.append(source)
                    valid_docs_count += 1
                except Exception as e:
                    error_count += 1
                    action.log(message="Error processing document", doc=doc, error=str(e))
                    # Continue processing the next document
                    continue

            result_msg = (
                f"Indexed {valid_docs_count} valid documents from {folder} with sources: {sources}. "
                f"Encountered {error_count} errors."
            )
            return result_msg

def run_rag_server(
    config: Optional[Path] = None,
    host: str = "0.0.0.0",
    port: int = 8088,
    workers: int = 1,
    title: str = "Just-Agent endpoint",
    section: Optional[str] = None,
    parent_section: Optional[str] = None,
    debug: bool = True,
    agents: Optional[Dict[str, BaseAgent]] = None,
) -> None:
    """Run the RAG server with the given configuration."""
    to_nice_stdout()

    # Initialize the API class with the updated configuration
    api = RAGServer(
        agent_config=config,
        agent_parent_section=parent_section,
        agent_section=section,
        debug=debug,
        title=title,
        agents=agents
    )
    
    uvicorn.run(
        api,
        host=host,
        port=port,
        workers=workers
    )


def run_rag_server_command(
    config: Optional[Path] = None,
    host: str = env_config.host,
    port: int = env_config.port,
    workers: int = env_config.workers,
    title: str = env_config.title,
    section: Optional[str] = env_config.section,
    parent_section: Optional[str] = env_config.parent_section,
    debug: bool = env_config.debug,
) -> None:
    """Run the FastAPI server for RAGServer with the given configuration."""
    agents = {"default": DEFAULT_RAG_AGENT} if config is None else None
    run_rag_server(
        config=config,
        host=host,
        port=port,
        workers=workers,
        title=title,
        section=section,
        parent_section=parent_section,
        debug=debug,
        agents=agents
    )

if __name__ == "__main__":
    env_config = RAGServerConfig()
    app = typer.Typer()
    app.command()(run_rag_server_command)
    app()