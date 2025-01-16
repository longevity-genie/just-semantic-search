from pathlib import Path
import sys
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

def index_folder(
    folder: Path,
    rag: MeiliRAG,
    splitter: SplitterType = SplitterType.SEMANTIC,
    model: EmbeddingModel = EmbeddingModel.GTE_LARGE
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

def index_file(
    filename: Path,
    abstract: str,
    title: str,
    source: str,
    host: str = "127.0.0.1",
    port: int = 7700,
    start_server: bool = True,
    model: EmbeddingModel = EmbeddingModel.JINA_EMBEDDINGS_V3,
    api_key: Optional[str] = None,
    meili_service_dir: Path = meili_service_dir,
    recreate_index: bool = True
) -> None:
    """Implementation function to index a single file."""
    with start_action(message_type="index_file", filename=str(filename)) as action:
        if action:
            action.log(message_type="processing_file", filename=str(filename))
        
        if api_key is None:
            api_key = os.getenv("MEILI_MASTER_KEY", "fancy_master_key")
        
        if start_server:
            if action:
                action.log(message_type="starting_server", host=host, port=port)
            ensure_meili_is_running(meili_service_dir, host, port)
        sentence_transformer_model = load_sentence_transformer_from_enum(model)
        
        if action:
            action.log(message_type="model_device", device=str(sentence_transformer_model.device))

        splitter = ArticleSemanticSplitter(model=sentence_transformer_model)
        documents = splitter.split_file(filename, embed=True, abstract=abstract, title=title, source=source)
        pprint(documents)

        rag = MeiliRAG(
            index_name="test",
            model=model,
            host=host,
            port=port,
            api_key=api_key,
            create_index_if_not_exists=True,
            recreate_index=recreate_index
        )
        
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