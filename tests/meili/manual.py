from just_semantic_search.splitter_factory import SplitterType
from just_semantic_search.text_splitters import *
from just_semantic_search.utils.logs import to_nice_file, to_nice_stdout
from just_semantic_search.embeddings import *
from just_semantic_search.utils.tokens import *
from pathlib import Path

import typer
import os
from dotenv import load_dotenv
from just_semantic_search.meili.rag import *
from just_semantic_search.meili.rag import *
import time
from pathlib import Path

from eliot._output import *
from eliot import start_task

from just_semantic_search.meili.utils.services import ensure_meili_is_running
from functions import create_meili_rag, index_file, index_folder
from test_cases import test_rsids, test_superhero_search

from tests.config import *

load_dotenv(override=True)
key = os.getenv("MEILI_MASTER_KEY", "fancy_master_key")

app = typer.Typer()

to_nice_file(json_log, rendered_file=rendered_log)
to_nice_stdout()


@app.command()
def test_search(
    index_name: str = typer.Option("tacutopapers", "--index-name", "-n"),
    model: EmbeddingModel = EmbeddingModel.GTE_LARGE,
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7700, "--port", "-p"),
    api_key: str = typer.Option(None, "--api-key", "-k"),
    ensure_server: bool = typer.Option(True, "--ensure-server", "-e", help="Ensure Meilisearch server is running"),
    score_threshold: float = typer.Option(0.8, "--score-threshold", "-s", help="Score threshold for hits"),
    tell_text: bool = typer.Option(False, "--tell-text", "-t", help="Tell text of hits"),
    ):

    if api_key is None:
        api_key = os.getenv("MEILI_MASTER_KEY", "fancy_master_key")

    transformer_model = load_sentence_transformer_from_enum(model)
    
    with start_task(action_type="test_search", 
                    index_name=index_name, model_name=model, host=host, port=port, api_key=api_key) as action:
        if ensure_server:
            action.log(message_type="ensuring_server", host=host, port=port)
            ensure_meili_is_running(meili_service_dir, host, port)
        config = MeiliConfig(host=host, port=port, api_key=api_key)
        rag = MeiliRAG(index_name, model, config, 
                    create_index_if_not_exists=True, 
                    recreate_index=False)
        result1 = test_rsids(rag, model=model, tell_text=tell_text, score_threshold=score_threshold)
        action.log(message_type="test_rsids_complete")
        result2 = test_superhero_search(rag, model=model, tell_text=tell_text, score_threshold=score_threshold)
        action.log(message_type="test_superhero_search_complete")


@app.command("index-folder")
def index_folder_command(
    index_name: str = typer.Option("tacutopapers", "--index-name", "-n"),
    model: EmbeddingModel = typer.Option(EmbeddingModel.GTE_LARGE.value, "--model", "-m", help="Embedding model to use"),
    folder: Path = typer.Option(tacutopapers_dir, "--folder", "-f", help="Folder containing documents to index"),
    splitter: SplitterType = typer.Option(SplitterType.SEMANTIC.value, "--splitter", "-s", help="Splitter type to use"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7700, "--port", "-p"),
    api_key: str = typer.Option(None, "--api-key", "-k"),
    skip_parsing: bool = typer.Option(False, "--skip-parsing", "-s"),
    test: bool = typer.Option(True, "--test", "-t", help="Test the index"),
    ensure_server: bool = typer.Option(False, "--ensure-seсrver", "-e", help="Ensure Meilisearch server is running")
) -> None:
    with start_task(action_type="index_folder", 
                    index_name=index_name, model_name=model, host=host, port=port, 
                    api_key=api_key, skip_parsing=skip_parsing, test=test, ensure_server=ensure_server) as action:
        rag = create_meili_rag(index_name, model, host, port, api_key, skip_parsing, ensure_server)
        transformer_model = load_sentence_transformer_from_enum(model)
        index_folder(folder, rag, splitter, model)    
        if test:
            test_rsids(rag, model=transformer_model)

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


@app.command("index-file")
def index_file_command(
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
    start_server: bool = typer.Option(True, "--start-server", "-s", help="Start Meilisearch server"),
    model: EmbeddingModel = typer.Option(EmbeddingModel.GTE_LARGE.value, "--model", "-m", help="Embedding model to use"),
):
    with start_task(action_type="index_file") as action:
        index_file(filename, abstract, title, source, host, port, start_server)

if __name__ == "__main__":
   app()
