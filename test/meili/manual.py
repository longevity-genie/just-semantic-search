from just_semantic_search.article_semantic_splitter import ArticleSemanticSplitter
from just_semantic_search.semanic_splitter import SemanticSplitter
from just_semantic_search.utils.logs import to_nice_file
from sentence_transformers import SentenceTransformer
from just_semantic_search.embeddings import *
from just_semantic_search.utils.tokens import *
from pathlib import Path
import time
#from just_semantic_search.utils import RenderingFileDestination


import typer
import os
from dotenv import load_dotenv
from just_semantic_search.meili.rag import *
from just_semantic_search.meili.rag import *
import time
from pathlib import Path
import subprocess
import requests
from eliot import log_call, to_file, log_message
import sys
from datetime import datetime

from eliot._output import *


load_dotenv(override=True)
key = os.getenv("MEILI_MASTER_KEY", "fancy_master_key")


app = typer.Typer()

current_dir = Path(__file__).parent
project_dir = current_dir.parent.parent  # Go up 3 levels from test/meili to project root
data_dir = project_dir / "data"
logs = project_dir / "logs"
tacutopapers_dir = data_dir / "tacutopapers_test_rsids_10k"

# Configure Eliot to output to both stdout and log files
log_file_path = logs / f"meili_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logs.mkdir(exist_ok=True)  # Ensure logs directory exists

# Create both JSON and rendered log files
json_log = open(f"{log_file_path}.json", "w")
rendered_log = open(f"{log_file_path}.txt", "w")


# Set up logging destinations
json_log = open(f"{log_file_path}.json", "w")
rendered_log = open(f"{log_file_path}.txt", "w")

# Set up logging destinations
to_file(sys.stdout)  # Keep console output
to_nice_file(json_log, rendered_log)


@log_call(action_type="ensure_meili_running", include_args=["host", "port"])
def ensure_meili_is_running(project_root: Path, host: str = "127.0.0.1", port: int = 7700) -> bool:
    """Start MeiliSearch container if not running and wait for it to be ready"""
    meili_script = project_root / "bin" / "meili.sh"
    url = f"http://{host}:{port}/health"
    
    # Check if MeiliSearch is already running
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
    except requests.exceptions.ConnectionError:
        pass

    # Start MeiliSearch in background
    process = subprocess.Popen(["/bin/bash", str(meili_script)], 
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    time.sleep(4)

    # Wait for MeiliSearch to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            time.sleep(1)
            continue
    
    raise RuntimeError("MeiliSearch failed to start")


@app.command()
@log_call(
    action_type="index_folder", 
    include_args=["index_name", "model_name", "host", "port", "skip_parsing"],
    include_result=False
)
def index_folder(
    index_name: str = typer.Option("tacutopapers", "--index-name", "-n"),
    model_name: str = typer.Option("gte-large", "--model-name", "-m"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7700, "--port", "-p"),
    api_key: str = typer.Option(None, "--api-key", "-k"),
    skip_parsing: bool = typer.Option(False, "--skip-parsing", "-s"),
) -> None:
    """Create and configure a MeiliRAG index."""
    if api_key is None:
        api_key = os.getenv("MEILI_MASTER_KEY", "fancy_master_key")

    if "gte" in model_name:
        model: SentenceTransformer = load_gte_large()
    else:
        raise ValueError(f"Model {model_name} not found!")

    config = MeiliConfig(host=host, port=port, api_key=api_key)
    rag = MeiliRAG(index_name, model_name, config, 
                   create_index_if_not_exists=True, 
                   recreate_index=not skip_parsing)

    if not skip_parsing:
        splitter = SemanticSplitter(model, batch_size=64, normalize_embeddings=False)
        documents = splitter.split_folder(tacutopapers_dir)

    rag.add_documents_sync(documents)
    dcs = rag.get_documents()
    
    log_message(
        message_type="index_folder_complete",
        index_name=index_name,
        document_count=len(dcs.results)
    )


@app.command()
@log_call(
    action_type="list_documents", 
    include_args=["host", "port", "index_name", "model_name"]
)
def documents(
    host: str = typer.Option("127.0.0.1", "--host", help="Meilisearch host"),
    port: int = typer.Option(7700, "--port", "-p", help="Meilisearch port"),
    index_name: str = typer.Option("tacutopapers", "--index-name", "-n", help="Name of the index to create"),
    model_name: str = typer.Option("gte-large", "--model-name", "-m", help="Name of the model to use"),
):
    ensure_meili_is_running(project_dir, host, port)
    rag = MeiliRAG(index_name, model_name, MeiliConfig(host=host, port=port, api_key=key), 
                   create_index_if_not_exists=True, recreate_index=False)
    dcs = rag.get_documents()
    log_message(message_type="documents_list", documents=dcs)


@app.command()
@log_call(
    action_type="index_file",
    include_args=["host", "port"],
    include_result=False
)
def index_file(
    filename: Path = typer.Option(
        tacutopapers_dir / "108.txt",
        "--filename", "-f",
        help="Path to the file to index"
    ),
    abstract: str = typer.Option(
        "Multiple studies characterizing the human ageing phenotype...",
        "--abstract", "-a",
        help="Abstract text for the document"
    ),
    title: str = typer.Option(
        "The Digital Ageing Atlas: integrating the diversity of age-related changes into a unified resource",
        "--title", "-t",
        help="Title of the document"
    ),
    source: str = typer.Option(
        "https://doi.org/10.1093/nar/gku843",
        "--source", "-s",
        help="Source URL or reference"
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Meilisearch host"),
    port: int = typer.Option(7700, "--port", "-p", help="Meilisearch port")
):
    """Main function to index a single file."""
    log_message(message_type="processing_file", filename=str(filename))
    ensure_meili_is_running(project_dir, host, port)
    
    model: SentenceTransformer = load_gte_large()
    log_message(message_type="model_device", device=str(model.device))

    splitter = ArticleSemanticSplitter(model)
    documents = splitter.split_file(filename, embed=True, abstract=abstract, title=title, source=source)

    config = MeiliConfig(host=host, port=port, api_key=key)
    rag = MeiliRAG("test", splitter.model_name, config, create_index_if_not_exists=True, recreate_index=True)
    
    # Ensure documents have vectors with the correct model name key
    for doc in documents:
        if splitter.model_name not in doc.vectors:
            log_message(message_type="warning", warning=f"Document missing vector for model {splitter.model_name}")
    
    rag.add_documents_sync(documents=documents)
    time.sleep(4)  # Add 4 second delay
    
    test = rag.get_documents()

if __name__ == "__main__":
   app()
