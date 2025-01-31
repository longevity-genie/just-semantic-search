from pathlib import Path
import subprocess
import sys
import threading
sys.path.append(str(Path(__file__).parent.parent.parent)) #stupid bug fix

from just_semantic_search.text_splitters import *
from just_semantic_search.embeddings import *
from just_semantic_search.utils.tokens import *
from pathlib import Path
import time

import os
from just_semantic_search.meili.rag import *
from just_semantic_search.meili.rag import *
import time

from eliot._output import *

from just_semantic_search.meili.utils.services import ensure_meili_is_running
from tests.config import *
from just_semantic_search.splitter_factory import SplitterType, create_splitter
from rich.pretty import pprint



"""
index_name: str = typer.Option("tacutopapers", "--index-name", "-n"),
    model: EmbeddingModel = typer.Option(EmbeddingModel.JINA_EMBEDDINGS_V3.value, "--model", "-m", help="Embedding model to use"),
    folder: Path = typer.Option(tacutopapers_dir, "--folder", "-f", help="Folder containing documents to index"),
    splitter: SplitterType = typer.Option(SplitterType.TEXT.value, "--splitter", "-s", help="Splitter type to use"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7700, "--port", "-p"),
    api_key: str = typer.Option(None, "--api-key", "-k"),
    skip_parsing: bool = typer.Option(False, "--skip-parsing", "-s"),
    test: bool = typer.Option(False, "--test", "-t", help="Test the index"),
    ensure_server: bool = typer.Option(False, "--ensure-seсrver", "-e", help="Ensure Meilisearch server is running"),
    recreate_index: bool = typer.Option(False, "--recreate-index", "-r", help="Recreate index")
"""

def index_folder(
    folder: Path,
    rag: MeiliRAG,
    splitter: SplitterType = SplitterType.SEMANTIC,
    model: EmbeddingModel = EmbeddingModel.JINA_EMBEDDINGS_V3
) -> None:
    """Index documents from a folder using the provided MeiliRAG instance."""
    with start_action(message_type="index_folder", folder=str(folder)) as action:
        sentence_transformer_model = load_sentence_transformer_from_enum(model)
        splitter_instance = create_splitter(splitter, sentence_transformer_model)
        documents = splitter_instance.split_folder(folder)
        rag.add_documents(documents)
        action.add_success_fields(
            message_type="index_folder_complete",
            index_name=rag.index_name,
            documents_added_count=len(documents)
        )


def create_meili_rag(
    index_name: str,
    model: EmbeddingModel = EmbeddingModel.JINA_EMBEDDINGS_V3,
    host: str = "127.0.0.1",
    port: int = 7700,
    api_key: Optional[str] = None,
    skip_parsing: bool = False,
    ensure_server: bool = False
) -> MeiliRAG:
    """Create and configure a MeiliRAG instance."""
    with start_action(message_type="creating_meili_rag", 
                      index_name=index_name, model_name=model, host=host, port=port, api_key=api_key, skip_parsing=skip_parsing, ensure_server=ensure_server) as action:
        if api_key is None:
            api_key = os.getenv("MEILI_MASTER_KEY", "fancy_master_key")
        
        if ensure_server:
            if action:
                action.log(message_type="ensuring_server", host=host, port=port)
            ensure_meili_is_running(meili_service_dir, host, port)
        
        return MeiliRAG(
            index_name=index_name,
            model=model,
            host=host,
            port=port,
            api_key=api_key,
            create_index_if_not_exists=True,
            recreate_index=not skip_parsing
        )

def index_file(
    rag: MeiliRAG,
    filename: Path,
    abstract: str,
    title: str,
    source: str,
    splitter_type: SplitterType = SplitterType.ARTICLE.value
) -> None:
    """Implementation function to index a single file."""
    with start_action(message_type="index_file", filename=str(filename)) as action:
        if action:
            action.log(message_type="processing_file", filename=str(filename))
        
        sentence_transformer_model = load_sentence_transformer_from_enum(rag.model)
        
        if action:
            action.log(message_type="model_device", device=str(sentence_transformer_model.device))

        splitter = create_splitter(splitter_type, sentence_transformer_model)
        documents = splitter.split_file(filename, embed=True, abstract=abstract, title=title, source=source)
        action.log("documents count", documents_count=len(documents))   

        # Ensure documents have vectors with the correct model name key
        for doc in documents:
            if splitter.model_name not in doc.vectors:
                if action:
                    action.log(message_type="warning", warning=f"Document missing vector for model {splitter.model_name}")
                    
        
        rag.add_documents(documents=documents)
        time.sleep(4)  # Add 4 second delay
        test = rag.get_documents()
        if action:
            action.log(message_type="documents in index count", count=len(test.results))

def simulate_meilisearch_disconnection(duration: int = 10):
    """
    Simulate a network disconnection for the MeiliSearch Docker container.

    :param duration: Duration in seconds to keep the container disconnected.
    """
    def disconnect_and_reconnect():
        container_name = "meilisearch"  # Matches the container_name in docker-compose.yaml
        network_name = "meili_meilisearch"  # Docker Compose prefixes the network with the directory name
        
        try:
            # Check if container exists and is running
            container_check = subprocess.run(["docker", "container", "inspect", container_name], 
                                          capture_output=True, check=False)
            if container_check.returncode != 0:
                print(f"Container {container_name} not found or not running")
                return

            # Check if network exists
            network_check = subprocess.run(["docker", "network", "inspect", network_name], 
                                        capture_output=True, check=False)
            if network_check.returncode != 0:
                print(f"Network {network_name} not found")
                return

            # Disconnect the container from the network
            disconnect_result = subprocess.run(
                ["docker", "network", "disconnect", network_name, container_name], 
                capture_output=True, check=False)
            
            if disconnect_result.returncode == 0:
                with start_action(action_type="network_disconnection") as action:
                    action.log(
                        message_type="disconnection_success",
                        container_name=container_name,
                        network_name=network_name
                    )
                time.sleep(duration)
            else:
                with start_action(action_type="network_disconnection") as action:
                    action.log(
                        message_type="disconnection_failure",
                        error=disconnect_result.stderr.decode()
                    )
                
        finally:
            # Try to reconnect the container to the network
            connect_result = subprocess.run(
                ["docker", "network", "connect", network_name, container_name],
                capture_output=True, check=False)
            
            if connect_result.returncode == 0:
                with start_action(action_type="network_reconnection") as action:
                    action.log(
                        message_type="reconnection_success",
                        container_name=container_name,
                        network_name=network_name
                    )
            else:
                with start_action(action_type="network_reconnection") as action:
                    action.log(
                        message_type="reconnection_failure",
                        error=connect_result.stderr.decode()
                    )

    # Run the disconnection and reconnection in a separate thread
    threading.Thread(target=disconnect_and_reconnect).start()
