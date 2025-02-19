from typing import List, Dict, Optional
from fastapi import Query
from pydantic import BaseModel
from just_agents.base_agent import BaseAgent
from just_agents.web.chat_ui_rest_api import ChatUIAgentRestAPI
from just_agents.web.rest_api import AgentRestAPI
from eliot import start_task

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
        self.post("/search", description="Perform semantic search with hybrid capabilities")(self.search)
        

    async def search(self, request: SearchRequest) -> List[Dict]:
        """
        Perform a semantic search with optional hybrid search capabilities.
        
        Args:
            request: SearchRequest containing search parameters
            
        Returns:
            List of matching documents with their metadata
        """
        with start_task as action:
            action.log("checking search request")
            # Here you would implement the actual search logic
            # This is a placeholder that should be implemented based on your specific RAG implementation
            raise NotImplementedError("Search functionality needs to be implemented")
