import os

import json
from pathlib import Path
import pprint
import warnings
import typer
from typing import Optional

# Suppress all warnings from torch, transformers and flash-attn
warnings.filterwarnings('ignore', message='.*flash_attn.*')

from dotenv import load_dotenv

from just_agents import llm_options
from just_agents.web.chat_ui_agent import ChatUIAgent
from just_agents.web.web_agent import WebAgent
from just_semantic_search.embeddings import EmbeddingModel
from just_semantic_search.meili.tools import all_indexes, search_documents, search_documents_debug
import os

from just_semantic_search.meili.utils.services import ensure_meili_is_running
load_dotenv(override=True)

current_dir = Path(__file__).parent
project_dir = current_dir.parent.parent.parent  # Go up 2 levels from test/meili to project root
data_dir = project_dir / "data"
logs = project_dir / "logs"
tacutopapers_dir = data_dir / "tacutopapers_test_rsids_10k"
meili_service_dir = project_dir

GEMINI_2_FLASH= {
    "model": "gemini/gemini-2.0-flash",
    "temperature": 0.0
}

def search_documents_no_db(query: str) -> str:
    """
    This functions makes a fency search
    query: str
    """
    return "42 is a meaning of life"

call_indexes = "YOU DO NOT search documents until you will retrive all the indexes in the database. When you search you are only alllowed to select from the indexes that you retrived, do not invent indexes!"

DEFAULT_RAG_AGENT = WebAgent(#ChatUIAgent(
        llm_options=GEMINI_2_FLASH,
        tools=[search_documents, all_indexes],
        system_prompt=f"""
        You are a helpful assistant that can search for documents in a MeiliSearch database. 
        f{call_indexes}
        You can only search indexes that you got from all_indexes function, do not invent indexes that do not exist.
        You MUST ALWAYS provide sources for all the documents. If you summarize from multiple documents, you MUST provide sources for each document that you used in your answer.
        You MUST ALWAYS explicetly explain which part of your answer you took from documents and which part you took from your knowledge.
        YOU NEVER CALL THE TOOL WITH THE SAME PARAMETERS MULTIPLE TIMES.
        The search document function uses semantic search.
        """
    )

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
    load_dotenv(override=True)
    
    if ensure_meili:
        ensure_meili_is_running(meili_dir)
    
    # Configure the agent
    llm_config = {
        "model": model,
        "temperature": temperature
    }
    indexes = all_indexes(non_empty=True)
    
    agent = WebAgent(
        llm_options=llm_config,
        tools=[search_documents, all_indexes],
        system_prompt=DEFAULT_RAG_AGENT.system_prompt.replace(call_indexes, f"You can only search indexes: {indexes}. NEVER put index parameter in the search function which is not in this list.")
    )
    
    
    if debug:
        agent.memory.add_on_message(lambda role, message: pprint.pprint(message))
    
    result = agent.query(prompt)
    print(result)

if __name__ == "__main__":
    app()
   