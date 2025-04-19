from pathlib import Path
import subprocess
import sys
import threading
sys.path.append(str(Path(__file__).parent.parent.parent)) #stupid bug fix

from just_semantic_search.splitters.text_splitters import *
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
from just_semantic_search.splitters.splitter_factory import SplitterType, create_splitter
from rich.pretty import pprint



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
        
        return MeiliRAG.get_instance(
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
        # Look for container names that contain 'meilisearch'
        container_name_cmd = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}", "--filter", "name=meilisearch"], 
            capture_output=True, text=True, check=False
        )
        
        if container_name_cmd.returncode != 0 or not container_name_cmd.stdout.strip():
            # Try with podman if docker fails
            container_name_cmd = subprocess.run(
                ["podman", "ps", "--format", "{{.Names}}", "--filter", "name=meilisearch"], 
                capture_output=True, text=True, check=False
            )
        
        container_names = container_name_cmd.stdout.strip().split('\n')
        container_name = container_names[0] if container_names and container_names[0] else "meilisearch"
        
        with start_action(action_type="container_detection") as action:
            action.log(message_type="container_name", container_name=container_name)
        
        # Get the network the container is connected to
        network_cmd = subprocess.run(
            ["docker", "inspect", "--format", "{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}", container_name],
            capture_output=True, text=True, check=False
        )
        
        if network_cmd.returncode != 0 or not network_cmd.stdout.strip():
            # Try with podman if docker fails
            network_cmd = subprocess.run(
                ["podman", "inspect", "--format", "{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}", container_name],
                capture_output=True, text=True, check=False
            )
        
        network_name = network_cmd.stdout.strip() if network_cmd.stdout.strip() else "meili_meilisearch"
        
        with start_action(action_type="network_detection") as action:
            action.log(message_type="network_name", network_name=network_name)
        
        try:
            # Check if container exists and is running
            container_check = subprocess.run(["docker", "container", "inspect", container_name], 
                                          capture_output=True, check=False)
            
            # Try with podman if docker fails
            if container_check.returncode != 0:
                container_check = subprocess.run(["podman", "container", "inspect", container_name], 
                                              capture_output=True, check=False)
            
            if container_check.returncode != 0:
                with start_action(action_type="network_disconnection") as action:
                    action.log(
                        message_type="container_not_found",
                        container_name=container_name
                    )
                return

            # Check if network exists
            network_check = subprocess.run(["docker", "network", "inspect", network_name], 
                                        capture_output=True, check=False)
            
            # Try with podman if docker fails
            if network_check.returncode != 0:
                network_check = subprocess.run(["podman", "network", "inspect", network_name], 
                                            capture_output=True, check=False)
            
            if network_check.returncode != 0:
                with start_action(action_type="network_disconnection") as action:
                    action.log(
                        message_type="network_not_found",
                        network_name=network_name
                    )
                return

            # Disconnect the container from the network
            disconnect_result = subprocess.run(
                ["docker", "network", "disconnect", network_name, container_name], 
                capture_output=True, check=False)
            
            # Try with podman if docker fails
            if disconnect_result.returncode != 0:
                disconnect_result = subprocess.run(
                    ["podman", "network", "disconnect", network_name, container_name], 
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
            
            # Try with podman if docker fails
            if connect_result.returncode != 0:
                connect_result = subprocess.run(
                    ["podman", "network", "connect", network_name, container_name],
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
