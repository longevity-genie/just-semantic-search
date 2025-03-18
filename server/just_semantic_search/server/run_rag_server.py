#!/usr/bin/env python
"""
Server runner module for Just Semantic Search RAG Server
This module provides a clean way to run the RAG server with multiple workers
by exposing the application as a module-level variable that can be imported.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict

import typer
import uvicorn
from eliot import start_task
from just_agents.base_agent import BaseAgent
from just_semantic_search.server.rag_server import RAGServer, RAGServerConfig
from just_semantic_search.server.rag_agent import default_rag_agent, default_annotation_agent

# Create a global application instance that can be imported by Uvicorn
# Will be populated by the create_app function
app = None

def create_app(
    agent_profiles: Optional[Path] = None,
    agent_parent_section: Optional[str] = None,
    agent_section: Optional[str] = None,
    debug: bool = True,
    title: str = "Just-Agent endpoint",
    description: str = "Welcome to the Just-Semantic-Search and Just-Agents API!",
    agents: Optional[Dict[str, BaseAgent]] = None,
    config: Optional[RAGServerConfig] = None
) -> RAGServer:
    """Create and configure a RAG server application instance."""
    if config is None:
        config = RAGServerConfig()
    
    return RAGServer(
        agent_profiles=agent_profiles,
        agent_parent_section=agent_parent_section,
        agent_section=agent_section,
        debug=debug,
        title=title,
        description=description,
        agents=agents,
        config=config
    )

def run_server_with_cli():
    """
    Run the server using command line arguments via Typer
    """
    cli_app = typer.Typer()
    cli_app.command()(run_server_command)
    cli_app()

def run_server_command(
    host: str = typer.Option("0.0.0.0", help="Host to bind the server to"),
    port: int = typer.Option(8091, help="Port to run the server on"),
    workers: int = typer.Option(1, help="Number of worker processes"),
    debug: bool = typer.Option(False, help="Enable debug mode")
) -> None:
    """Run the RAG server with the given configuration."""
    with start_task(action_type="run_rag_server_command", workers=workers) as action:
        # Create default agents
        agents = {
            "rag_agent": default_rag_agent(), 
            "annotation_agent": default_annotation_agent()
        }
        
        action.log(f"Starting server with {workers} workers")
        
        # If running with multiple workers, use the import string approach
        if workers > 1:
            # Set up server configuration first
            config = RAGServerConfig()
            config.set_general_port(port)
            config.host = host
            config.debug = debug
            
            # Export the configuration as environment variables so 
            # the worker processes can access it
            os.environ["JUST_SERVER_PORT"] = str(port)
            os.environ["JUST_SERVER_HOST"] = host
            os.environ["JUST_SERVER_DEBUG"] = "1" if debug else "0"
            
            action.log(f"Using module:app pattern for multi-worker setup")
            
            # Run using the module:app pattern
            uvicorn.run(
                "just_semantic_search.server.run_rag_server:get_app",
                host=host,
                port=port,
                workers=workers
            )
        else:
            # For a single worker, create and run the app directly
            config = RAGServerConfig()
            config.set_general_port(port)
            config.host = host
            config.debug = debug
            
            app = create_app(agents=agents, config=config)
            
            action.log(f"Running with single worker")
            
            uvicorn.run(
                app,
                host=host,
                port=port
            )

def get_app():
    """
    Function that returns the app instance for workers
    This is called by Uvicorn when using the module:app format
    """
    # Read configuration from environment variables
    port = int(os.environ.get("JUST_SERVER_PORT", "8091"))
    host = os.environ.get("JUST_SERVER_HOST", "0.0.0.0")
    debug = os.environ.get("JUST_SERVER_DEBUG", "0") == "1"
    
    # Create and return the application
    config = RAGServerConfig()
    config.set_general_port(port)
    config.host = host
    config.debug = debug
    
    from just_semantic_search.server.rag_agent import default_rag_agent, default_annotation_agent
    agents = {
        "rag_agent": default_rag_agent(), 
        "annotation_agent": default_annotation_agent()
    }
    
    return create_app(agents=agents, config=config)

if __name__ == "__main__":
    run_server_with_cli() 