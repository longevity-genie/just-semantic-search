from just_semantic_search.article_splitter import ArticleSplitter
from typing import List, Optional
from just_semantic_search.meili.utils.services import ensure_meili_is_running
from just_semantic_search.server.agentic_indexing import AgenticIndexing
from just_semantic_search.server.utils import default_annotation_agent, get_project_directories, load_environment_files
from pydantic import BaseModel, Field
from just_semantic_search.text_splitters import *
from just_semantic_search.embeddings import *

from just_semantic_search.utils.tokens import *
from pathlib import Path
from just_agents import llm_options

import typer
import os
from just_semantic_search.meili.rag import *
from pathlib import Path

from eliot._output import *
from eliot import start_task


from pathlib import Path
from pycomfort import files
from eliot import start_task


app = typer.Typer()

@app.command("index-markdown")
def index_markdown_command(
    folder: Path = typer.Argument(..., help="Folder containing documents to index"),
    index_name: str = typer.Option(..., "--index-name", "-i", "-n"),
    model: EmbeddingModel = typer.Option(EmbeddingModel.JINA_EMBEDDINGS_V3.value, "--model", "-m", help="Embedding model to use"),
    host: str = typer.Option(None, "--host", help="Meilisearch host (defaults to env MEILI_HOST or 127.0.0.1)"),
    port: int = typer.Option(None, "--port", "-p", help="Meilisearch port (defaults to env MEILI_PORT or 7700)"),
    characters_limit: int = typer.Option(None, "--characters-limit", "-c", help="Characters limit for text processing"),
    max_seq_length: int = typer.Option(None, "--max-seq-length", "-s", help="Maximum sequence length for text splitting"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="Meilisearch API key"),
    indexing_api_key: Optional[str] = typer.Option(None, "--indexing-api-key", help="API key for securing indexing operations"),
    ensure_server: bool = typer.Option(False, "--ensure-server", "-e", help="Ensure Meilisearch server is running"),
    recreate_index: bool = typer.Option(None, "--recreate-index", "-r", help="Recreate index"),
    depth: int = typer.Option(None, "--depth", "-d", help="Depth of folder parsing"),
    extensions: List[str] = typer.Option(None, "--extensions", "-x", help="File extensions to include"),
) -> None:
    # Load environment variables from .env files
    load_environment_files()
    
    # Get project directories
    dirs = get_project_directories()
    meili_service_dir = dirs["meili_service_dir"]
    
    # Use environment values as defaults if parameters weren't provided
    if host is None:
        host = os.getenv("MEILI_HOST", "127.0.0.1")
    if port is None:
        port = int(os.getenv("MEILI_PORT", "7700"))
    if api_key is None:
        api_key = os.getenv("MEILI_MASTER_KEY", "fancy_master_key")
    if indexing_api_key is None:
        indexing_api_key = os.getenv("INDEXING_API_KEY")
    if characters_limit is None:
        characters_limit = int(os.getenv("INDEX_CHARACTERS_FOR_ABSTRACT", "10000"))
    if max_seq_length is None:
        max_seq_length = int(os.getenv("INDEX_MAX_SEQ_LENGTH", "3600"))
    if recreate_index is None:
        recreate_index = os.getenv("PARSING_RECREATE_MEILI_INDEX", "False").lower() in ("true", "1", "yes")
    if depth is None:
        depth = int(os.getenv("INDEX_DEPTH", "1"))
    if extensions is None:
        extensions_str = os.getenv("INDEX_EXTENSIONS", ".md")
        extensions = extensions_str.split(",") if "," in extensions_str else [extensions_str]
    
    # If indexing API key is provided, check it against environment
    env_indexing_api_key = os.getenv("INDEXING_API_KEY")
    if env_indexing_api_key and (not indexing_api_key or indexing_api_key != env_indexing_api_key):
        print("Error: Invalid or missing API key for indexing operations")
        return
    
    with start_task(action_type="index_markdown", 
                    index_name=index_name, model_name=str(model), host=host, port=port, 
                    api_key=api_key, ensure_server=ensure_server) as action:
        # Ensure Meilisearch is running if requested
        if ensure_server:
            ensure_meili_is_running(meili_service_dir, host, port)
        
        # Create RAG instance
        rag = MeiliRAG.get_instance(
            index_name=index_name,
            model=model,
            host=host,
            port=port,
            api_key=api_key,
            create_index_if_not_exists=True,
            recreate_index=recreate_index
        )
        
        # Create indexing instance and index the folder
        indexing = AgenticIndexing(
            annotation_agent=default_annotation_agent(),
            embedding_model=model
        )
        
        indexing.index_md_txt(
            rag=rag,
            folder=Path(folder),
            max_seq_length=max_seq_length,
            characters_for_abstract=characters_limit,
            depth=depth,
            extensions=extensions)
        action.log(message_type="indexing_complete", index_name=index_name)
        

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # If no arguments provided, show help
        sys.argv.append("--help")
    app(prog_name="index-markdown")