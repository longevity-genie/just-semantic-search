import pprint
from just_semantic_search.splitter_factory import SplitterType, create_splitter
from just_semantic_search.text_splitters import *
from pycomfort.logging import to_nice_file, to_nice_stdout
from just_semantic_search.embeddings import *
from just_semantic_search.utils.tokens import *
from pathlib import Path

import typer
import os
from dotenv import load_dotenv
from just_semantic_search.meili.rag import *
import time
from pathlib import Path

from eliot._output import *
from eliot import start_task

from just_semantic_search.meili.utils.services import ensure_meili_is_running

from datetime import datetime
from pathlib import Path
from typing import List
from pycomfort.logging import to_nice_file, to_nice_stdout

current_dir = Path(__file__).parent
project_dir = current_dir.parent.parent.parent  # Go up 2 levels from test/meili to project root
data_dir = project_dir / "data"
logs = project_dir / "logs"
tacutopapers_dir = data_dir / "tacutopapers_test_rsids_10k"
meili_service_dir = project_dir / "meili"

# Configure Eliot to output to both stdout and log files
log_file_path = logs / f"manual_meili_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logs.mkdir(exist_ok=True)  # Ensure logs directory exists

# Create both JSON and rendered log files
json_log = open(f"{log_file_path}.json", "w")
rendered_log = open(f"{log_file_path}.txt", "w")


load_dotenv(override=True)
to_nice_file(json_log, rendered_file=rendered_log)
to_nice_stdout()
key = os.getenv("MEILI_MASTER_KEY", "fancy_master_key")

app = typer.Typer()

to_nice_file(json_log, rendered_file=rendered_log)
to_nice_stdout()

@app.command()
def embedders(
    index_name: str = typer.Option("tacutopapers", "--index-name", "-n"),
    model: EmbeddingModel = typer.Option(EmbeddingModel.JINA_EMBEDDINGS_V3, "--model", "-m", help="Embedding model to use"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7700, "--port", "-p"),
    api_key: str = typer.Option(None, "--api-key", "-k"),
    ensure_server: bool = typer.Option(True, "--ensure-server", "-e", help="Ensure Meilisearch server is running")
    ):

    if api_key is None:
        api_key = os.getenv("MEILI_MASTER_KEY", "fancy_master_key")
    if ensure_server:
        ensure_meili_is_running(meili_service_dir, host, port)

    rag = MeiliRAG(
        index_name=index_name,
        model=model,
        host=host,
        port=port,
        api_key=api_key,
        create_index_if_not_exists=True,
        recreate_index=False
    )
    print("Embedders:")
    pprint(rag.index.get_embedders())


@app.command("index-folder")
def index_folder_command(
    index_name: str = typer.Option(os.getenv("MEILI_INDEX_NAME", "tacutopapers"), "--index-name", "-n"),
    model: EmbeddingModel = typer.Option(EmbeddingModel.JINA_EMBEDDINGS_V3.value, "--model", "-m", help="Embedding model to use"),
    folder: Path = typer.Option(..., "--folder", "-f", help="Folder containing documents to index"),
    splitter: SplitterType = typer.Option(SplitterType.TEXT.value, "--splitter", "-s", help="Splitter type to use"),
    host: str = typer.Option(os.getenv("MEILI_HOST", "127.0.0.1"), "--host"),
    port: int = typer.Option(os.getenv("MEILI_PORT", 7700), "--port", "-p"),
    api_key: Optional[str] = typer.Option(os.getenv("MEILI_MASTER_KEY", "fancy_master_key"), "--api-key", "-k"),
    ensure_server: bool = typer.Option(False, "--ensure-server", "-e", help="Ensure Meilisearch server is running"),
    recreate_index: bool = typer.Option(os.getenv("MEILI_RECREATE_INDEX", False), "--recreate-index", "-r", help="Recreate index")
) -> None:
    with start_task(action_type="index_folder", 
                    index_name=index_name, model_name=str(model), host=host, port=port, 
                    api_key=api_key, ensure_server=ensure_server) as action:
        if api_key is None:
            api_key = os.getenv("MEILI_MASTER_KEY", "fancy_master_key")
        if ensure_server:
            ensure_meili_is_running(meili_service_dir, host, port)
        
        rag = MeiliRAG(
            index_name=index_name,
            model=model,
            host=host,
            port=port,
            api_key=api_key,
            create_index_if_not_exists=True,
            recreate_index=recreate_index
        )
        rag.index_folder(Path(folder), splitter, model)


@app.command()
def documents(
    host: str = typer.Option(os.getenv("MEILI_HOST", "127.0.0.1"), "--host", help="Meilisearch host"),
    port: int = typer.Option(os.getenv("MEILI_PORT", 7700), "--port", "-p", help="Meilisearch port"),
    index_name: str = typer.Option("tacutopapers", "--index-name", "-n", help="Name of the index to create"),
    model: EmbeddingModel = typer.Option(EmbeddingModel.JINA_EMBEDDINGS_V3, "--model", "-m", help="Embedding model to use"),
    ensure_server: bool = typer.Option(True, "--ensure-server", "-e", help="Ensure Meilisearch server is running")
):
    with start_task(action_type="documents") as action:
        if ensure_server:
            ensure_meili_is_running(meili_service_dir, host, port)
            rag = MeiliRAG(
                index_name=index_name,
                model=model,
                host=host,
                port=port,
                api_key=key,
                create_index_if_not_exists=True,
                recreate_index=False
            )
            info = rag.get_documents()
            action.log(message_type="documents_list", count = len(info.results))


@app.command()
def delete_index(
    index_names: List[str] = typer.Argument(
        None,
        help="Names of the indexes to delete (space-separated list)"
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Meilisearch host"),
    port: int = typer.Option(7700, "--port", "-p", help="Meilisearch port"),
    api_key: str = typer.Option(None, "--api-key", "-k", help="Meilisearch API key"),
    model: EmbeddingModel = typer.Option(EmbeddingModel.JINA_EMBEDDINGS_V3, "--model", "-m", help="Embedding model to use"),
    ensure_server: bool = typer.Option(True, "--ensure-server", "-e", help="Ensure Meilisearch server is running")
):
    if api_key is None:
        api_key = os.getenv("MEILI_MASTER_KEY", "fancy_master_key")
    with start_task(action_type="delete_index") as action:

        if ensure_server:
            ensure_meili_is_running(meili_service_dir, host, port)
        
        for index_name in index_names:
            rag = MeiliRAG(
                index_name=index_name,
                model=model,
                host=host,
                port=port,
                api_key=api_key,
                create_index_if_not_exists=True,
                recreate_index=False
            )
            rag.delete_index()
        action.log(message_type="delete_index_complete", index_names=index_names)



if __name__ == "__main__":
    app(prog_name="meili-exec", help=True)  # Show help by default
