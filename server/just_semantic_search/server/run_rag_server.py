

from pathlib import Path
from typing import Optional
from just_agents.web.config import ChatUIAgentConfig
from just_agents.web.run_agent import run_agent_server
from just_semantic_search.server.rag_agent import DEFAULT_RAG_AGENT
from just_semantic_search.utils.logs import to_nice_stdout
import typer
from just_semantic_search.server.rag_server import RAGServer

env_config = ChatUIAgentConfig()
app = typer.Typer()

from pathlib import Path
from typing import Optional, Type
from just_agents.web.config import ChatUIAgentConfig
from just_agents.web.rest_api import AgentRestAPI
from just_agents.web.chat_ui_rest_api import ChatUIAgentRestAPI
import uvicorn
import typer

from pycomfort.logging import to_nice_stdout
from eliot import start_action, start_task
from just_agents.base_agent import BaseAgent
from typing import Dict

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
    to_nice_stdout()

    # Initialize the API class with the updated configuration
    api = RAGServer(
        agent_parent_section=parent_section,
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


@app.command()
def run_rag_server_command(
    config: Optional[Path] = typer.Argument(
        None,
        help="Path to the YAML configuration file. Defaults to 'agent_profiles.yaml' in current directory"
    ),
    host: str = typer.Option(env_config.host, help="Host to bind the server to"),
    port: int = typer.Option(env_config.port, help="Port to run the server on"),
    workers: int = typer.Option(env_config.workers, help="Number of worker processes"),
    title: str = typer.Option(env_config.title, help="Title for the API endpoint"),
    section: Optional[str] = typer.Option(env_config.section, help="Optional section name in the config file"),
    parent_section: Optional[str] = typer.Option(env_config.parent_section, help="Optional parent section name in the config file"),
    debug: bool = typer.Option(env_config.debug, help="Debug mode"),

) -> None:
    """Run the FastAPI server for ChatUIAgentRestAPI with the given configuration."""
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
    app()