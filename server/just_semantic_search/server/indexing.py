from just_semantic_search.article_splitter import ArticleSplitter
from typing import List, Optional
from just_semantic_search.meili.utils.services import ensure_meili_is_running
from just_semantic_search.server.utils import default_annotation_agent, get_project_directories, load_environment_files
from pydantic import BaseModel, Field
from just_semantic_search.text_splitters import *
from just_semantic_search.embeddings import *

from just_semantic_search.utils.tokens import *
from pathlib import Path
from just_agents import llm_options
from just_agents.base_agent import BaseAgent

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

class Annotation(BaseModel):
    abstract: str
    authors: List[str] = Field(default_factory=list)
    title: str
    source: str
    
    model_config = {
        "extra": "forbid",
        "arbitrary_types_allowed": True
    }

class Indexing(BaseModel):
    
    annotation_agent: BaseAgent
    embedding_model: EmbeddingModel

    def clean_agent_history(self):
        system_messages = [msg for msg in self.annotation_agent.memory.messages if msg.get("role") == "system"]
        self.annotation_agent.memory.clear_messages()  # Clear all messages
        self.annotation_agent.memory.add_messages(system_messages)  # Add back only system messages

    def _process_single_paper(self, f: Path, rag: MeiliRAG, max_seq_length: int, characters_for_abstract: int) -> List[dict]:
        """Process a single paper file and add it to the RAG index.
        
        Args:
            f: Path to the file
            rag: MeiliRAG instance for document storage
            max_seq_length: Maximum sequence length for chunks
            characters_for_abstract: Number of characters to use for extracting metadata
            
        Returns:
            List of document chunks created from this paper
        """
        with start_task(message_type="process_paper", file=str(f.name)) as file_task:
            text = f.read_text()[:characters_for_abstract]
            paper = None
            
            with start_task(message_type="process_paper.annotation") as annotation_task:
                enforce_validation = os.environ.get("INDEXING_ENFORCE_VALIDATION", "False").lower() in ("true", "1", "yes")
                query = f"Extract the abstract, authors and title of the following paper (from file {f.name}):\n{text}"
                try:
                    annotation_task.log(self.annotation_agent.llm_options)
                    response = self.annotation_agent.query_structural(
                            query, 
                            Annotation, 
                            enforce_validation=enforce_validation, 
                            remember_query=False)
                    # Keep only system messages in memory
                    
                    paper = Annotation.model_validate(response)
                    annotation_task.log(message_type="process_paper.annotation_complete", title=paper.title)
                except Exception as e:
                    annotation_task.log(message_type="process_paper.annotation_error", 
                                       error=str(e), 
                                       error_type=str(type(e).__name__), query=query)
                    # Re-raise the exception to maintain original behavior
                    raise
            self.clean_agent_history()
            with start_task(message_type="process_paper.splitting") as splitting_task:
                splitter_instance = ArticleSplitter(model=rag.sentence_transformer, max_seq_length=max_seq_length)
                docs = splitter_instance.split(text, title=paper.title, abstract=paper.abstract, authors=paper.authors, source=paper.source)
                splitting_task.log(message_type="process_paper.splitting_complete", chunks_count=len(docs))
            
            with start_task(message_type="process_paper.embedding_and_indexing") as embedding_task:
                rag.add_documents(docs)
                embedding_task.log(message_type="process_paper.embedding_complete", chunks_count=len(docs))
            
            file_task.log(message_type="process_paper.indexed", document_count=len(docs))
            return docs

    def index_md_txt(self, rag: MeiliRAG, folder: Path, 
                     max_seq_length: Optional[int] = 3600, 
                     characters_for_abstract: int = 10000, depth: int = -1, extensions: List[str] = [".md", ".txt"]
                     ) -> List[dict]:
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
            fs = files.traverse(folder, lambda x: x.suffix in extensions, depth=depth)
            documents = []
            
            for f in fs:
                try:
                    paper_docs = self._process_single_paper(f, rag, max_seq_length, characters_for_abstract)
                    documents.extend(paper_docs)
                except Exception as e:
                    task.log(message_type="index_markdown.paper_processing_error", 
                             file=str(f.name),
                             error=str(e),
                             error_type=str(type(e).__name__))
                    # Continue processing other papers
                    continue
            
            task.add_success_fields(
                message_type="index_markdown_complete",
                index_name=rag.index_name,
                documents_added_count=len(documents)
            )
            return documents


    def index_markdown_tool(self, folder: Path, index_name: str,) -> List[dict]:
        model_str = os.getenv("EMBEDDING_MODEL", EmbeddingModel.JINA_EMBEDDINGS_V3.value)
        model = EmbeddingModel(model_str)

        max_seq_length: Optional[int] = os.getenv("INDEX_MAX_SEQ_LENGTH", 3600)
        characters_for_abstract: int = os.getenv("INDEX_CHARACTERS_FOR_ABSTRACT", 5000)
        
        # Create and return RAG instance with conditional recreate_index
        # It should use default environment variables for host, port, api_key, create_index_if_not_exists, recreate_index
        rag = MeiliRAG(
            index_name=index_name,
            model=model,        # The embedding model used for the search
        )
        return self.index_md_txt(rag, folder, max_seq_length, characters_for_abstract)
    

    
    def index_markdown_folder(self, folder: str, index_name: str) -> str:
        """
        Indexes a folder with markdown files. The server should have access to the folder.
        Uses defensive checks for documents that might be either dicts or Document instances.
        Reports errors to Eliot logs without breaking execution; problematic documents are skipped.
        """
        
        with start_task(action_type="rag_server_index_markdown_folder", folder=folder, index_name=index_name) as action:
            folder_path = Path(folder)
            if not folder_path.exists():
                msg = f"Folder {folder} does not exist or the server does not have access to it"
                action.log(msg)
                return msg
            
            with start_task(action_type="rag_server_index_markdown_folder.config") as config_task:
                model_str = os.getenv("EMBEDDING_MODEL", EmbeddingModel.JINA_EMBEDDINGS_V3.value)
                model = EmbeddingModel(model_str)

                max_seq_length: Optional[int] = os.getenv("INDEX_MAX_SEQ_LENGTH", 3600)
                characters_for_abstract: int = os.getenv("INDEX_CHARACTERS_FOR_ABSTRACT", 10000)
                config_task.log(message_type="config_loaded", model=model_str, max_seq_length=max_seq_length)
            
            # Create and return RAG instance with conditional recreate_index
            with start_task(action_type="rag_server_index_markdown_folder.create_rag") as rag_task:
                rag = MeiliRAG(
                    index_name=index_name,
                    model=model,        # The embedding model used for the search
                )
                rag_task.log(message_type="rag_created", index_name=index_name)
            
            with start_task(action_type="rag_server_index_markdown_folder.indexing") as indexing_task:
                docs = self.index_md_txt(rag, folder_path, max_seq_length, characters_for_abstract)
                indexing_task.log(message_type="indexing_complete", docs_count=len(docs))
            
            sources = []
            valid_docs_count = 0
            error_count = 0

            with start_task(action_type="rag_server_index_markdown_folder.validation") as validation_task:
                for doc in docs:
                    try:
                        if isinstance(doc, dict):
                            source = doc.get("source")
                            if source is None:
                                raise ValueError(f"Document (dict) missing 'source' key: {doc}")
                        elif isinstance(doc, Document):
                            source = getattr(doc, "source", None)
                            if source is None:
                                raise ValueError(f"Document instance missing 'source' attribute: {doc}")
                        else:
                            raise TypeError(f"Unexpected document type: {type(doc)} encountered in documents list")

                        sources.append(source)
                        valid_docs_count += 1
                    except Exception as e:
                        error_count += 1
                        validation_task.log(message="Error processing document", doc=str(doc)[:100], error=str(e))
                        # Continue processing the next document
                        continue
                
                validation_task.log(message_type="validation_complete", valid_count=valid_docs_count, error_count=error_count)

            result_msg = (
                f"Indexed {valid_docs_count} valid documents from {folder} with sources: {sources}. "
                f"Encountered {error_count} errors."
            )
            return result_msg


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
    
    with start_task(action_type="index_markdown", 
                    index_name=index_name, model_name=str(model), host=host, port=port, 
                    api_key=api_key, ensure_server=ensure_server) as action:
        # Ensure Meilisearch is running if requested
        if ensure_server:
            ensure_meili_is_running(meili_service_dir, host, port)
        
        # Create RAG instance
        rag = MeiliRAG(
            index_name=index_name,
            model=model,
            host=host,
            port=port,
            api_key=api_key,
            create_index_if_not_exists=True,
            recreate_index=recreate_index
        )
        
        # Create indexing instance and index the folder
        indexing = Indexing(
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