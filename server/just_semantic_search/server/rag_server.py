from typing import List, Dict, Optional
from fastapi import Query
from just_semantic_search.meili.tools import search_documents
from pydantic import BaseModel
from just_agents.base_agent import BaseAgent
from just_agents.web.chat_ui_rest_api import ChatUIAgentRestAPI
from just_agents.web.rest_api import AgentRestAPI
from eliot import start_task
from just_semantic_search.server.rag_agent import DEFAULT_RAG_AGENT

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

    def __init__(self, agents: Optional[Dict[str, BaseAgent]] = None, *args, **kwargs):
        if agents is not None:
            kwargs["agents"] = agents
        super().__init__(*args, **kwargs)
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