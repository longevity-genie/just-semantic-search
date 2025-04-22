from pprint import pprint
from just_semantic_search.splitters.splitter_factory import SplitterType
from just_semantic_search.splitters.text_splitters import *
from pycomfort.logging import to_nice_file, to_nice_stdout
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
def embedders(
    index_name: str = typer.Option("tacutopapers", "--index-name", "-n"),
    model: EmbeddingModel = typer.Option(EmbeddingModel.JINA_EMBEDDINGS_V3.value, "--model", "-m", help="Embedding model to use"),
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
      

@app.command()
def test_search(
    index_name: str = typer.Option("tacutopapers", "--index-name", "-n"),
    model: EmbeddingModel = typer.Option(EmbeddingModel.JINA_EMBEDDINGS_V3.value, "--model", "-m", help="Embedding model to use"),
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
        rag = MeiliRAG(
            index_name=index_name,
            model=model,
            host=host,
            port=port,
            api_key=api_key,
            create_index_if_not_exists=True,
            recreate_index=False
        )
        result1 = test_rsids(rag, transformer_model=transformer_model, tell_text=tell_text, score_threshold=score_threshold)
        action.log(message_type="test_rsids_complete")
        result2 = test_superhero_search(rag, transformer_model=transformer_model, tell_text=tell_text, score_threshold=score_threshold)
        action.log(message_type="test_superhero_search_complete")


