from just_semantic_search.article_splitter import ArticleSplitter
from typing import List, Optional
from pydantic import BaseModel
from just_semantic_search.text_splitters import *
from pycomfort.logging import to_nice_file, to_nice_stdout
from just_semantic_search.embeddings import *

from just_semantic_search.utils.tokens import *
from pathlib import Path
from just_agents import llm_options
from just_agents.base_agent import BaseAgent

import typer
import os
from dotenv import load_dotenv
from just_semantic_search.meili.rag import *
from pathlib import Path

from eliot._output import *
from eliot import start_task

from just_semantic_search.meili.utils.services import ensure_meili_is_running

from datetime import datetime
from pathlib import Path
from pycomfort import files
from eliot import start_task

current_dir = Path(__file__).parent
project_dir = Path(os.getenv("APP_DIR", str(current_dir.parent.parent.parent))).absolute()   # Go up 2 levels from test/meili to project root
data_dir = project_dir / os.getenv("DATA_DIR", "data")
logs = project_dir / os.getenv("LOG_DIR", "logs")
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

class Annotation(BaseModel):
    abstract: str
    authors: List[str] = Field(default_factory=list)
    title: str
    source: str
    
    model_config = {
        "extra": "forbid",
        "arbitrary_types_allowed": True
    }


def index_markdown(rag: MeiliRAG, folder: Path, max_seq_length: Optional[int] = 3600, characters_for_abstract: int = 10000, depth: int = -1, extensions: List[str] = [".md"],
                   options: Optional[llm_options.LLMOptions] = None) -> List[dict]:
    """
    Index markdown files from a folder into MeiliSearch.
    
    Args:
        rag: MeiliRAG instance for document storage and retrieval
        folder: Path to the folder containing markdown files
        characters_limit: Maximum number of characters to process per file
        
    Returns:
        List of processed documents
    """
    with start_task(message_type="index_markdown", folder=str(folder)) as task:

        fs = files.traverse(folder, lambda x: x.suffix in extensions)
        
        
        splitter_instance = ArticleSplitter(model=rag.sentence_transformer, max_seq_length=max_seq_length)

        options = options or llm_options.OPENAI_GPT4oMINI

        agent = BaseAgent(  # type: ignore
            llm_options=options,
            system_prompt="""
            You are a paper annotator. You extract the abstract, authors and titles of the papers.
            Abstract and authors must be exactly he way they are in the paper, do not edit them.
            You provide your output as json object of the following JSON format:
            {
                "abstract": "...",
                "authors": ["...", "..."],
                "title": "...",
                "source": "...",
            }
            For string either use one line or use proper escape characters (\n) for line breaks
            Make sure to provide the output in the correct format, do not add any other text or comments.
            For source you either give DOI, pubmed or filename (if doi or pubmed is not available).
            File filename you give a filename of the file in the folder together with the extension.
            """)
        fs = files.files(folder)
        documents = []
        for f in fs:
            text = f.read_text()[:characters_for_abstract]
            response = agent.query_structural(
                f"Extract the abstract, authors and title of the following paper (from file {f.name}:\n {text}",
                Annotation
            )
            paper = Annotation.model_validate(response)
            docs = splitter_instance.split(text, title=paper.title, abstract=paper.abstract, authors=paper.authors, source=paper.source)
            rag.add_documents(docs)
            documents.extend(docs)
            task.log(message_type="index_markdown_document.indexed", document_count=len(documents))
        
        task.add_success_fields(
            message_type="index_markdown_complete",
            index_name=rag.index_name,
            documents_added_count=len(documents)
        )
        return documents
    


def index_markdown_tool(folder: Path, index_name: str,) -> List[dict]:
    model_str = os.getenv("EMBEDDING_MODEL", EmbeddingModel.JINA_EMBEDDINGS_V3.value)
    model = EmbeddingModel(model_str)

    max_seq_length: Optional[int] = os.getenv("INDEX_MAX_SEQ_LENGTH", 3600)
    characters_for_abstract: int = os.getenv("INDEX_CHARACTERS_FOR_ABSTRACT", 10000)
    
    # Create and return RAG instance with conditional recreate_index
    # It should use default environment variables for host, port, api_key, create_index_if_not_exists, recreate_index
    rag = MeiliRAG(
        index_name=index_name,
        model=model,        # The embedding model used for the search
    )
    return index_markdown(rag, folder, max_seq_length, characters_for_abstract)

@app.command("index-markdown")
def index_markdown_command(
    folder: Path = typer.Argument(..., help="Folder containing documents to index"),
    index_name: str = typer.Option(..., "--index-name", "-i", "-n"),
    model: EmbeddingModel = typer.Option(EmbeddingModel.JINA_EMBEDDINGS_V3.value, "--model", "-m", help="Embedding model to use"),
    host: str = typer.Option(os.getenv("MEILI_HOST", "127.0.0.1"), "--host"),
    port: int = typer.Option(os.getenv("MEILI_PORT", 7700), "--port", "-p"),
    characters_limit: int = typer.Option(10000, "--characters-limit", "-c", help="Characters limit to use"),
    max_seq_length: int = typer.Option(3600, "--max-seq-length", "-s", help="Maximum sequence length for text splitting"),
    api_key: Optional[str] = typer.Option(os.getenv("MEILI_MASTER_KEY", "fancy_master_key"), "--api-key", "-k"),
    ensure_server: bool = typer.Option(False, "--ensure-server", "-e", help="Ensure Meilisearch server is running"),
    recreate_index: bool = typer.Option(os.getenv("PARSING_RECREATE_MEILI_INDEX", False), "--recreate-index", "-r", help="Recreate index"),
    depth: int = typer.Option(1, "--depth", "-d", help="Depth of folder parsing"),
    extensions: List[str] = typer.Option([".md"], "--extensions", "-x", help="File extensions to include"),
) -> None:
    with start_task(action_type="index_markdown", 
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
        index_markdown(rag, Path(folder), max_seq_length, characters_limit, depth, extensions)
        

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # If no arguments provided, show help
        sys.argv.append("--help")
    app(prog_name="index-markdown")
