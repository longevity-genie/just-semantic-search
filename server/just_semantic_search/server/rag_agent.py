import pprint
import warnings
import typer
from typing import Optional
from pathlib import Path

# Suppress all warnings from torch, transformers and flash-attn
warnings.filterwarnings('ignore', message='.*flash_attn.*')

from just_agents.web.web_agent import WebAgent
from just_semantic_search.meili.tools import search_documents, all_indexes
from just_semantic_search.server.utils import (
    load_environment_files, 
    get_project_directories, 
    default_rag_agent,
    default_annotation_agent,
    setup_meili
)

# Load environment variables
load_environment_files()

# Get project directories
dirs = get_project_directories()
meili_service_dir = dirs["meili_service_dir"]
tacutopapers_dir = dirs["data_dir"] / "tacutopapers_test_rsids_10k"

app = typer.Typer()

@app.command()
def query_agent(
    prompt: str = typer.Argument(default="Which machine learning models are used for CGM?", help="The question to ask the agent"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug output with pprint"),
    model: str = typer.Option("gemini/gemini-2.0-flash", "--model", "-m", help="LLM model to use"),
    temperature: float = typer.Option(0.0, "--temperature", "-t", help="Temperature for LLM generation"),
    ensure_meili: bool = typer.Option(True, "--ensure-meili/--no-ensure-meili", help="Ensure MeiliSearch is running"),
    meili_dir: Optional[Path] = typer.Option(
        meili_service_dir, 
        "--meili-dir",
        help="Directory containing MeiliSearch service",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
):
    """
    Query the RAG agent with a prompt and get a response.
    """
    # Reload environment variables
    load_environment_files()
    
    # Setup MeiliSearch and get available indexes
    indexes = setup_meili(meili_dir, ensure_meili)
    
    # Configure the agent
    llm_config = {
        "model": model,
        "temperature": temperature
    }
    
    rag_agent = default_rag_agent()
    
    # Create call_indexes replacement text
    call_indexes_replacement = f"You can only search indexes: {indexes}. NEVER put index parameter in the search function which is not in this list."
    
    # Find the call_indexes text in the system prompt
    call_indexes = "YOU DO NOT search documents until you will retrive all the indexes in the database. When you search you are only alllowed to select from the indexes that you retrived, do not invent indexes!"
    
    agent = WebAgent(
        llm_options=llm_config,
        tools=[search_documents, all_indexes],
        system_prompt=rag_agent.system_prompt.replace(call_indexes, call_indexes_replacement)
    )
    
    if debug:
        agent.memory.add_on_message(lambda role, message: pprint.pprint(message))
    
    result = agent.query(prompt)
    print(result)

if __name__ == "__main__":
    print("Starting RAG agent")
    app()

   