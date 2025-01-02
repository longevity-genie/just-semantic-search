from just_semantic_search.text_splitters import *
from just_semantic_search.utils.logs import to_nice_file, to_nice_stdout
from just_semantic_search.utils.models import get_sentence_transformer_model_name
from sentence_transformers import SentenceTransformer
from just_semantic_search.embeddings import *
from just_semantic_search.utils.tokens import *
from pathlib import Path
import time

import typer
import os
from dotenv import load_dotenv
from just_semantic_search.meili.rag import *
from just_semantic_search.meili.rag import *
import time
from pathlib import Path
from datetime import datetime

from eliot._output import *
from eliot import start_task

from just_semantic_search.meili.utils.services import ensure_meili_is_running
from test_cases import test_rsids, test_superhero_search


load_dotenv(override=True)
key = os.getenv("MEILI_MASTER_KEY", "fancy_master_key")



app = typer.Typer()

current_dir = Path(__file__).parent
project_dir = current_dir.parent.parent  # Go up 3 levels from test/meili to project root
data_dir = project_dir / "data"
logs = project_dir / "logs"
tacutopapers_dir = data_dir / "tacutopapers_test_rsids_10k"
meili_service_dir = project_dir / "services" / "meili"

# Configure Eliot to output to both stdout and log files
log_file_path = logs / f"manual_meili_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logs.mkdir(exist_ok=True)  # Ensure logs directory exists

# Create both JSON and rendered log files
json_log = open(f"{log_file_path}.json", "w")
rendered_log = open(f"{log_file_path}.txt", "w")


# Set up logging destinations
json_log = open(f"{log_file_path}.json", "w")
rendered_log = open(f"{log_file_path}.txt", "w")


# Set up logging destinations
#to_file(sys.stdout)  # Keep console output
to_nice_file(json_log, rendered_file=rendered_log)
to_nice_stdout()



@app.command()
def test_search(
    index_name: str = typer.Option("tacutopapers", "--index-name", "-n"),
    model_name: str = typer.Option("gte-large", "--model-name", "-m"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7700, "--port", "-p"),
    api_key: str = typer.Option(None, "--api-key", "-k"),
    ensure_server: bool = typer.Option(True, "--ensure-server", "-e", help="Ensure Meilisearch server is running"),
    score_threshold: float = typer.Option(0.8, "--score-threshold", "-s", help="Score threshold for hits"),
    tell_text: bool = typer.Option(False, "--tell-text", "-t", help="Tell text of hits"),
    ):

    if api_key is None:
        api_key = os.getenv("MEILI_MASTER_KEY", "fancy_master_key")

    if "gte" in model_name:
        model: SentenceTransformer = load_gte_large()
        model_name = get_sentence_transformer_model_name(model)
    else:
        raise ValueError(f"Model {model_name} not found!")
    
    with start_task(action_type="test_search", 
                    index_name=index_name, model_name=model_name, host=host, port=port, api_key=api_key) as action:
        if ensure_server:
            action.log(message_type="ensuring_server", host=host, port=port)
            ensure_meili_is_running(meili_service_dir, host, port)
        config = MeiliConfig(host=host, port=port, api_key=api_key)
        rag = MeiliRAG(index_name, model_name, config, 
                    create_index_if_not_exists=True, 
                    recreate_index=False)
        result1 = test_rsids(rag, model=model, tell_text=tell_text, score_threshold=score_threshold)
        action.log(message_type="test_rsids_complete")
        result2 = test_superhero_search(rag, model=model, tell_text=tell_text, score_threshold=score_threshold)
        action.log(message_type="test_superhero_search_complete")

@app.command()
def index_folder(
    index_name: str = typer.Option("tacutopapers", "--index-name", "-n"),
    model_name: str = typer.Option("gte-large", "--model-name", "-m"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7700, "--port", "-p"),
    api_key: str = typer.Option(None, "--api-key", "-k"),
    skip_parsing: bool = typer.Option(False, "--skip-parsing", "-s"),
    test: bool = typer.Option(True, "--test", "-t", help="Test the index"),
    ensure_server: bool = typer.Option(True, "--ensure-server", "-e", help="Ensure Meilisearch server is running")
) -> None:
    """Create and configure a MeiliRAG index."""
    if api_key is None:
        api_key = os.getenv("MEILI_MASTER_KEY", "fancy_master_key")

    if "gte" in model_name:
        model: SentenceTransformer = load_gte_large()
        model_name = get_sentence_transformer_model_name(model)
    else:
        raise ValueError(f"Model {model_name} not found!")
    
    with start_task(action_type="index_folder", 
                    index_name=index_name, model_name=model_name, host=host, port=port, api_key=api_key, skip_parsing=skip_parsing, test=test, ensure_server=ensure_server) as action:
        if ensure_server:
            action.log(message_type="ensuring_server", host=host, port=port)
            ensure_meili_is_running(meili_service_dir, host, port)
        splitter = SemanticSplitter(model=model, batch_size=64, normalize_embeddings=False)
        config = MeiliConfig(host=host, port=port, api_key=api_key)
        rag = MeiliRAG(index_name, splitter.model_name, config, 
                    create_index_if_not_exists=True, 
                    recreate_index=not skip_parsing)

        if not skip_parsing:
            documents = splitter.split_folder(tacutopapers_dir)
            rag.add_documents(documents)
        
        action.add_success_fields(
            message_type="index_folder_complete",
            index_name=index_name,
            documents_added_count=len(documents)
        )
        if test:
            test_rsids(rag, model=model)


@app.command()
def documents(
    host: str = typer.Option("127.0.0.1", "--host", help="Meilisearch host"),
    port: int = typer.Option(7700, "--port", "-p", help="Meilisearch port"),
    index_name: str = typer.Option("tacutopapers", "--index-name", "-n", help="Name of the index to create"),
    model_name: str = typer.Option("gte-large", "--model-name", "-m", help="Name of the model to use"),
):
    with start_task(action_type="documents") as action:
            ensure_meili_is_running(meili_service_dir, host, port)
            rag = MeiliRAG(index_name, model_name, MeiliConfig(host=host, port=port, api_key=key), 
                    create_index_if_not_exists=True, recreate_index=False)
            info = rag.get_documents()
            action.log(message_type="documents_list", count = len(info.results))


@app.command()
def delete_index(
    index_names: list[str] = typer.Option(
        ["tacutopapers", "test"], "--index-name", "-n", 
        help="Names of the indexes to delete (can specify multiple times)"
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Meilisearch host"),
    port: int = typer.Option(7700, "--port", "-p", help="Meilisearch port"),
    api_key: str = typer.Option(None, "--api-key", "-k", help="Meilisearch API key"),
    model_name: str = typer.Option("gte-large", "--model-name", "-m", help="Name of the model to use"),
    ensure_server: bool = typer.Option(True, "--ensure-server", "-e", help="Ensure Meilisearch server is running")
):
    if api_key is None:
        api_key = os.getenv("MEILI_MASTER_KEY", "fancy_master_key")
    with start_task(action_type="delete_index") as action:

        if ensure_server:
            ensure_meili_is_running(meili_service_dir, host, port)
        
        config = MeiliConfig(host=host, port=port, api_key=api_key)
        for index_name in index_names:
            rag = MeiliRAG(index_name, model_name, config, 
                        create_index_if_not_exists=True, 
                        recreate_index=False)
            rag.delete_index()
        action.log(message_type="delete_index_complete", index_names=index_names)

@app.command()
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
    port: int = typer.Option(7700, "--port", "-p", help="Meilisearch port"),
    start_server: bool = typer.Option(True, "--start-server", "-s", help="Start Meilisearch server")
):
    with start_task(action_type="index_file") as action: 
        """Main function to index a single file."""
        action.log(message_type="processing_file", filename=str(filename))
        if start_server:
            action.log(message_type="starting_server", host=host, port=port)
            ensure_meili_is_running(meili_service_dir, host, port)
        
        model: SentenceTransformer = load_gte_large()
        action.log(message_type="model_device", device=str(model.device))


        splitter = ArticleSemanticSplitter(model=model)
        documents = splitter.split_file(filename, embed=True, abstract=abstract, title=title, source=source)

        config = MeiliConfig(host=host, port=port, api_key=key)
        rag = MeiliRAG("test", splitter.model_name, config, create_index_if_not_exists=True, recreate_index=True)
        
        # Ensure documents have vectors with the correct model name key
        for doc in documents:
            if splitter.model_name not in doc.vectors:
                action.log(message_type="warning", warning=f"Document missing vector for model {splitter.model_name}")
        
        rag.add_documents(documents=documents)
        time.sleep(4)  # Add 4 second delay
        test = rag.get_documents()
        action.log(message_type="documents in index count", count = len(test.results))

if __name__ == "__main__":
   app()
