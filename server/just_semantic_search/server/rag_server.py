from typing import List, Dict, Optional
from fastapi import Query
from just_semantic_search.meili.tools import search_documents, all_indexes
from pydantic import BaseModel
from just_agents.base_agent import BaseAgent
from just_agents.web.rest_api import AgentRestAPI
from eliot import start_task
from just_semantic_search.server.rag_agent import DEFAULT_RAG_AGENT
from pathlib import Path
import uvicorn
from just_agents.web.config import ChatUIAgentConfig
import typer
from pycomfort.logging import to_nice_stdout

env_config = ChatUIAgentConfig()
app = typer.Typer()

class SearchRequest(BaseModel):
    query: str
    semantic_ratio: float = Query(default=0.5, ge=0.0, le=1.0)
    limit: int = Query(default=10, ge=1)
    offset: int = Query(default=0, ge=0)
    attributes: Optional[List[str]] = None
    embedding_model: str = "default"
    reranker: Optional[str] = None


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
        self._configure_rag_routes()

    def _configure_rag_routes(self):
        """Configure RAG-specific routes"""
        self.post("/search", description="Perform semantic search")(self.search)
        self.post("/search_agent", description="Perform advanced RAG-based search")(self.search_advanced)

    async def search(self, query: str, index: str, limit: int = 10) -> List[Dict]:
        """
        Perform a semantic search.
        
        Args:
            query: The search query string
            index: The index to search in
            limit: Maximum number of results to return (default: 10)
            
        Returns:
            List of matching documents with their metadata
        """
        with start_task(action_type="rag_server_search", query=query, index=index, limit=limit) as action:
            action.log("performing search")
            return search_documents(
                query=query,
                index=index,
                limit=limit
            )

    async def search_advanced(self, query: str) -> str:
        """
        Perform an advanced search using the RAG agent that can provide contextual answers.
        
        Args:
            query: The search query string
            
        Returns:
            A detailed response from the RAG agent incorporating retrieved documents
        """
        with start_task(action_type="rag_server_advanced_search", query=query) as action:
            action.log("performing advanced RAG search")
            result = DEFAULT_RAG_AGENT.query(query)
            return result
    
    async def list_indexes(self, non_empty: bool = True) -> List[Dict]:
        """
        Get all indexes.
        """
        return all_indexes(non_empty=non_empty)

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
    env_config = ChatUIAgentConfig()
    app = typer.Typer()
    app.command()(run_rag_server_command)
    app()