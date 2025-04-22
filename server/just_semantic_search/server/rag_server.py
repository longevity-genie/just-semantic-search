from functools import cached_property
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from just_semantic_search.embeddings import EmbeddingModel
from just_semantic_search.meili.rag import MeiliRAG
from just_semantic_search.meili.tools import search_documents, all_indexes
from just_semantic_search.server.rag_agent import default_annotation_agent, default_rag_agent
from just_semantic_search.splitters.splitter_factory import SplitterType
from pydantic import BaseModel, Field
from fastapi import Body, UploadFile
from just_agents.base_agent import BaseAgent
from just_agents.web.chat_ui_rest_api import ChatUIAgentRestAPI, ChatUIAgentConfig
from eliot import start_task
from pathlib import Path
import typer
import uvicorn
from just_semantic_search.server.agentic_indexing import AgenticIndexing
from pathlib import Path
from just_semantic_search.server.utils import load_environment_files
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import RedirectResponse
from starlette.requests import Request
from starlette.responses import Response


class RAGServerConfig(ChatUIAgentConfig):
    """Configuration for the RAG server"""

    
    host: str = Field(
        default_factory=lambda: os.getenv("APP_HOST", "0.0.0.0").split()[0],
        description="Host address to bind the server to",
        examples=["0.0.0.0", "127.0.0.1"]
    )

    embedding_model: EmbeddingModel = Field(
        default=EmbeddingModel.JINA_EMBEDDINGS_V3,
        description="Embedding model to use"
    )
    
    api_key: Optional[str] = Field(
        default=None,
        description="API key for securing indexing operations"
    )

    def set_general_port(self, port: int):
        self.agent_port = port
        self.port = port



class SearchRequest(BaseModel):
    """Request model for basic semantic search"""
    query: str = Field(example="Glucose predictions models for CGM")
    index: str = Field(example="glucosedao")
    limit: int = Field(default=10, ge=1, example=30)
    semantic_ratio: float = Field(default=0.5, ge=0.0, le=1.0, example=0.5)
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Glucose predictions models for CGM",
                    "index": "glucosedao",
                    "limit": 30,
                    "semantic_ratio": 0.5
                }
            ]
        }
    }
   

class SearchAgentRequest(BaseModel):
    """Request model for RAG-based advanced search"""
    query: str = Field(example="Glucose predictions models for CGM")
    index: Optional[str] = Field(default=None, example="glucosedao")
    additional_instructions: Optional[str] = Field(default=None, example="You must always provide quotes from evidence followed by the sources (not in the end but immediately after the quote)")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Glucose predictions models for CGM",
                    "index": "glucosedao",
                    "additional_instructions": "You must always provide quotes from evidence followed by the sources (not in the end but immediately after the quote)"
                },
                {
                    "query": "Time series forecasting for glucose",
                    "index": None,
                    "additional_instructions": None
                }
            ]
        }
    }

class DeleteBySourceRequest(BaseModel):
    """Request model for deleting documents by source"""
    index_name: str = Field(example="glucosedao")
    source: str = Field(example="paper1.md")
    api_key: Optional[str] = Field(default=None, description="API key for securing indexing operations")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "index_name": "glucosedao",
                    "source": "paper1.md",
                    "api_key": None
                }
            ]
        }
    }

class IndexFolderRequest(BaseModel):
    """Request model for indexing a markdown folder"""
    folder: str = Field(example="/path/to/folder")
    index_name: str = Field(example="glucosedao")
    extensions: Optional[List[str]] = Field(default=None, example=[".md", ".txt"])
    api_key: Optional[str] = Field(default=None, description="API key for securing indexing operations")
    splitter: Optional[SplitterType] = Field(default=SplitterType.ARTICLE, description="Splitter to use for indexing")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "folder": "/data/glucosedao",
                    "index_name": "glucosedao",
                    "extensions": [".md", ".txt"],
                    "splitter": "article"
                },
                {
                    "folder": "/data/lifespan_json/posts_flat_blog",
                    "index_name": "lifespan",
                    "extensions": [".json"],
                    "splitter": "flat_json"
                }
            ]
        }
    }

class IndexFileRequest(BaseModel):
    """Request model for indexing a file (shared fields)"""
    index_name: str = Field(example="glucosedao")
    max_seq_length: Optional[int] = Field(default=3600, example=3600)
    abstract: Optional[str] = Field(default=None, example="This is a summary of the document")
    title: Optional[str] = Field(default=None, example="Document Title")
    source: Optional[str] = Field(default=None, example="source.txt")
    autoannotate: bool = Field(default=False, example=True)
    api_key: Optional[str] = Field(default=None, description="API key for securing indexing operations")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "index_name": "glucosedao",
                    "max_seq_length": 3600,
                    "abstract": "This is a summary of the document",
                    "title": "Document Title",
                    "source": "source.txt",
                    "autoannotate": True,
                    "api_key": None
                }
            ]
        }
    }

class IndexPDFRequest(IndexFileRequest):
    """Request model for indexing a PDF file"""
    mistral_api_key: Optional[str] = Field(default=None, description="API key for Mistral OCR service")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "index_name": "glucosedao",
                    "max_seq_length": 3600,
                    "abstract": "This is a summary of the PDF document",
                    "title": "PDF Document Title",
                    "source": "document.pdf",
                    "autoannotate": True,
                    "mistral_api_key": None,
                    "api_key": None
                }
            ]
        }
    }

class IndexJsonFilesRequest(BaseModel):
    """Request model for indexing JSON files with fully custom fields"""
    folder: str = Field(example="/path/to/json_files")
    index_name: str = Field(example="custom_docs")
    content_field: str = Field(example="content", description="Field name in JSON containing the document content (required for splitting)")
    max_seq_length: Optional[int] = Field(default=3600, example=3600, description="Maximum sequence length for chunks")
    extension: str = Field(default=".json", example=".json", description="File extension to look for")
    depth: int = Field(default=-1, example=-1, description="Depth of folder traversal (-1 for unlimited)")
    required_fields: Optional[List[str]] = Field(default=None, example=["id", "type"], description="Optional list of field names that must be present in each document")
    api_key: Optional[str] = Field(default=None, description="API key for securing indexing operations")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "folder": "/data/lifespan_json/posts_flat_blog",
                    "index_name": "lifespan",
                    "content_field": "content",
                    "max_seq_length": 3600,
                    "extension": ".json",
                    "depth": -1,
                    "required_fields": ["id", "type"],
                    "api_key": None
                }
            ]
        }
    }

class DeleteIndexRequest(BaseModel):
    """Request model for deleting an entire index"""
    index_name: str = Field(example="glucosedao", description="Name of the index to delete")
    api_key: Optional[str] = Field(default=None, description="API key for securing indexing operations")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "index_name": "test",
                    "api_key": None
                }
            ]
        }
    }

class RAGServer(ChatUIAgentRestAPI):
    """Extended REST API implementation that adds RAG (Retrieval-Augmented Generation) capabilities"""

    indexing: AgenticIndexing

    @cached_property
    def rag_agent(self):
        if "rag_agent" in self.agents:
            return self.agents["rag_agent"]
        elif "default" in self.agents:
            return self.agents["default"]
        else:
            raise ValueError("RAG agent not found")

    @cached_property
    def annotation_agent(self):
        if "annotation_agent" in self.agents:
            return self.agents["annotation_agent"]
        elif "annotator" in self.agents:
            return self.agents["annotator"]
        else:
            raise ValueError("Annotation agent not found")


    def __init__(self, 
                 agents: Optional[Dict[str, BaseAgent]] = None,
                 agent_profiles: Optional[Path | str] = None,
                 agent_section: Optional[str] = None,
                 agent_parent_section: Optional[str] = None,
                 debug: bool = False,
                 title: str = "Just-Semantic-Search and Just-Agents endpoint, go to /docs for more information about REST API",
                 description: str = "Welcome to the Just-Semantic-Search and Just-Agents API! <br><br>Explore the complete API documentation in your browser by visiting <a href='/docs'>/docs</a>. <br><br>There you can: <ul><li>Run agentic LLM completions</li><li>Index documents with Meilisearch</li><li>Perform semantic searches</li><li>Upload and process various document types</li></ul>",
                 config: Optional[RAGServerConfig] = None,
                 *args, **kwargs):
        if agents is not None:
            kwargs["agents"] = agents

        self.config = RAGServerConfig() if config is None else config
        super().__init__(
            agent_config=agent_profiles,
            agent_section=agent_section,
            agent_parent_section=agent_parent_section,
            debug=debug,
            title=title,
            description=description,
            *args, **kwargs
        )
        self.indexing = AgenticIndexing(
            annotation_agent=self.annotation_agent,
            embedding_model=config.embedding_model
        )
        self._indexes = None
        self._configure_rag_routes()
        
        # Add a middleware to handle the root route with highest priority
        class RootRedirectMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                if request.url.path == "/":
                    return RedirectResponse(url="/docs", status_code=307)
                return await call_next(request)
        
        # Add the middleware to the application
        self.add_middleware(RootRedirectMiddleware) #ugly way to redirect to docs as other ways failed
        
        default_index = os.getenv("DEFAULT_INDEX")
        if default_index is not None:
            with start_task(action_type="rag_server_set_default_index") as action:
                action.log("preloading default index", index=default_index)
                rag = MeiliRAG.get_instance(index_name=default_index)

    def _prepare_model_jsons(self):
        with start_task(action_type="rag_server_prepare_model_jsons") as action:
            action.log("PREPARING MODEL JSONS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            super()._prepare_model_jsons()
        
    def _initialize_config(self):
        """Overriding initialization from config"""
        with start_task(action_type="rag_server_initialize_config") as action:
            action.log(f"Config: {self.config}")
            
            # Use the shared utility function
            env_loaded = load_environment_files(self.config.env_keys_path)
            
            # Continue with the rest of the initialization
            if not Path(self.config.models_dir).exists():
                action.log(f"Creating models directory {self.config.models_dir} which does not exist")
                Path(self.config.models_dir).mkdir(parents=True, exist_ok=True)
            if "env/" in self.config.env_models_path:
                if not Path("env").exists():
                    action.log(f"Creating env directory {self.config.env_models_path} which does not exist")
                    Path("env").mkdir(parents=True, exist_ok=True)
            
                    

    @property
    def indexes(self) -> List[str]:
        """Lazy property that returns cached list of indexes or fetches them if not cached"""
        if self._indexes is None:
            self._indexes = self.list_indexes()
        return self._indexes
    

    def _configure_rag_routes(self):
        """Configure RAG-specific routes"""
        # Add a check to prevent duplicate route registration
        route_paths = [route.path for route in self.routes]
        
        # DON'T register the root route here anymore
        # We'll handle it directly in __init__
        
        if "/search" not in route_paths:
            self.post("/search", tags=["Search Operations"], description="Perform semantic search")(self.search)
        
        if "/search_agent" not in route_paths:
            self.post("/search_agent", tags=["Search Operations"], description="Perform advanced RAG-based search")(self.search_agent)
        
        if "/list_indexes" not in route_paths:
            self.post("/list_indexes", tags=["Indexes Operations"], description="Get all indexes")(self.list_indexes)
        
        if "/index_folder" not in route_paths:
            @self.post("/index_folder", tags=["Upload Operations"], description="Index a folder, by default indexes md and txt files")
            def index_folder(request: IndexFolderRequest = Body()):
                return self.indexing.index_folder(
                    folder=request.folder, 
                    index_name=request.index_name,
                    api_key=request.api_key,
                    extensions=request.extensions,
                    splitter=request.splitter
                )
        
        if "/upload_markdown_folder" not in route_paths:
            @self.post("/upload_markdown_folder", tags=["Upload Operations"], description="Upload a folder with markdown files")
            async def upload_markdown_folder(uploaded_file: UploadFile, index_name: str, api_key: Optional[str] = None):
                return self.indexing.index_upload_markdown_folder(
                    uploaded_file=uploaded_file,
                    index_name=index_name,
                    api_key=api_key
                )
        
        # Add new routes for PDF and text file upload
        if "/upload_pdf" not in route_paths:
            @self.post("/upload_pdf", tags=["Upload Operations"], description="Upload and index a PDF file")
            async def upload_pdf(
                file: UploadFile, 
                request: IndexPDFRequest
            ):
                return self.indexing.index_pdf_file(
                    file=file,
                    index_name=request.index_name,
                    max_seq_length=request.max_seq_length,
                    abstract=request.abstract,
                    title=request.title,
                    source=request.source,
                    autoannotate=request.autoannotate,
                    mistral_api_key=request.mistral_api_key,
                    api_key=request.api_key
                )
        
        if "/upload_text" not in route_paths:
            @self.post("/upload_text", tags=["Upload Operations"], description="Upload and index a text file")
            async def upload_text(
                file: UploadFile, 
                request: IndexFileRequest
            ):
                return self.indexing.index_text_file(
                    file=file,
                    index_name=request.index_name,
                    max_seq_length=request.max_seq_length,
                    abstract=request.abstract,
                    title=request.title,
                    source=request.source,
                    autoannotate=request.autoannotate,
                    api_key=request.api_key
                )

        if "/delete_by_source" not in route_paths:
            @self.post("/delete_by_source", tags=["Delete Operations"], description="Delete documents by their sources")
            def delete_by_source(request: DeleteBySourceRequest):
                return self.indexing.delete_by_source(
                    index_name=request.index_name,
                    source=request.source,
                    api_key=request.api_key
                )

        if "/delete_index" not in route_paths:
            @self.post("/delete_index", tags=["Indexes Operations", "Delete Operations"], description="Delete an entire index")
            def delete_index(request: DeleteIndexRequest):
                return self.indexing.delete_index(
                    index_name=request.index_name,
                    api_key=request.api_key
                )

        if "/index_json_files" not in route_paths:
            @self.post("/index_json_files", tags=["Upload Operations"], description="Index JSON files with custom fields")
            def index_json_files(request: IndexJsonFilesRequest):
                return self.indexing.index_json_files(
                    folder=request.folder,
                    index_name=request.index_name,
                    content_field=request.content_field,
                    max_seq_length=request.max_seq_length,
                    api_key=request.api_key,
                    extension=request.extension,
                    depth=request.depth,
                    required_fields=request.required_fields
                )

    

    def search(self, request: SearchRequest) -> list[str]:
        """
        Perform a semantic search.
        
        Args:
            request: SearchRequest object containing search parameters
            
        Returns:
            List of matching documents with their metadata
        """
        import time
        start_time = time.time()
        
        with start_task(action_type="rag_server_search", 
                       query=request.query, 
                       index=request.index, 
                       limit=request.limit) as action:
            action.log(f"Search method entered, time since request: {time.time() - start_time:.2f}s")
            
            # Log before search_documents call
            pre_search_time = time.time()
            action.log(f"About to perform search, time so far: {pre_search_time - start_time:.2f}s")
            
            results = search_documents(
                query=request.query,
                index=request.index,
                limit=request.limit,
                semantic_ratio=request.semantic_ratio
            )
            
            # Log after search_documents call
            post_search_time = time.time()
            action.log(f"Search completed in {post_search_time - pre_search_time:.2f}s, total time: {post_search_time - start_time:.2f}s")
            
            return results

    def search_agent(self, request: SearchAgentRequest) -> str:
        """
        Perform an advanced search using the RAG agent that can provide contextual answers.
        
        Args:
            request: SearchAgentRequest object containing the query, optional index, and additional instructions
            
        Returns:
            A detailed response from the RAG agent incorporating retrieved documents
        """

        with start_task(action_type="rag_server_advanced_search", query=request.query) as action:
            import uuid
            request_id = str(uuid.uuid4())[:8]
            action.log(f"[{request_id}] Received search_agent request")
            
            indexes = self.indexes if request.index is None else [request.index]
            query = f"Search the following query:```\n{request.query}\n```\nYou can only search in the following indexes: {indexes}"
            if request.additional_instructions is not None:
                query += f"\nADDITIONAL INSTRUCTIONS: {request.additional_instructions}"
            
            action.log(f"[{request_id}] Querying RAG agent")
            result = self.rag_agent.query(query)
            action.log(f"[{request_id}] Completed search_agent request")
            return result
    
    def list_indexes(self, non_empty: bool = True) -> List[str]:
        """
        Get all indexes and update the cache.
        """
        self._indexes = all_indexes(non_empty=non_empty)
        return self._indexes
    
    

    def root_endpoint(self):
        """Redirect to the API documentation"""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/docs")

def run_rag_server(
    agent_profiles: Optional[Path] = None,
    host: str = "0.0.0.0",
    port: int = 8091,
    workers: int = 1,
    title: str = "Just-Agent endpoint",
    description: str = "Welcome to the Just-Semantic-Search and Just-Agents API! <br><br>Explore the complete API documentation in your browser by visiting <a href='/docs'>/docs</a>. <br><br>There you can: <ul><li>Run agentic LLM completions</li><li>Index documents with Meilisearch</li><li>Perform semantic searches</li><li>Upload and process various document types</li></ul>",
    section: Optional[str] = None,
    parent_section: Optional[str] = None,
    debug: bool = True,
    agents: Optional[Dict[str, BaseAgent]] = None,
    api_key: Optional[str] = None
) -> None:
    """Run the RAG server with the given configuration."""
    # Initialize the API class with the updated configuration
    config = RAGServerConfig()
    config.set_general_port(port)
    
    # Set the API key from parameter or environment variable
    if api_key:
        config.api_key = api_key
    else:
        config.api_key = os.getenv("INDEXING_API_KEY")
    
    # If API key is provided, set it in environment for other components
    if config.api_key:
        os.environ["INDEXING_API_KEY"] = config.api_key

    api = RAGServer(
        agent_profiles=agent_profiles,
        agent_parent_section=parent_section,
        agent_section=section,
        debug=debug,
        title=title,
        description=description,
        agents=agents,
        config=config
    )
    
    uvicorn.run(
        api,
        host=host,
        port=port,
        workers=workers
    )
