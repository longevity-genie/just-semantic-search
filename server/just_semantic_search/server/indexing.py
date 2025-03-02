from just_semantic_search.article_splitter import ArticleSplitter
from typing import List, Optional
from pydantic import BaseModel
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
            
            
            splitter_instance = ArticleSplitter(model=rag.sentence_transformer, max_seq_length=max_seq_length)

            
            fs = files.files(folder)
            documents = []
            for f in fs:
                text = f.read_text()[:characters_for_abstract]
                enforce_validation = "gemini" in self.annotation_agent.llm_options["model"]
                response = self.annotation_agent.query_structural(
                        f"Extract the abstract, authors and title of the following paper (from file {f.name}):\n{text}", Annotation, enforce_validation=enforce_validation)
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


    def index_markdown_tool(self, folder: Path, index_name: str,) -> List[dict]:
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
        return self.index_md_txt(rag, folder, max_seq_length, characters_for_abstract)
    


    def index_markdown_tool(self, folder: Path, index_name: str,) -> List[dict]:
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
            docs = self.index_md_txt(rag, folder, max_seq_length, characters_for_abstract)
            sources = []
            valid_docs_count = 0
            error_count = 0

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
                    action.log(message="Error processing document", doc=doc, error=str(e))
                    # Continue processing the next document
                    continue

            result_msg = (
                f"Indexed {valid_docs_count} valid documents from {folder} with sources: {sources}. "
                f"Encountered {error_count} errors."
            )
            return result_msg

"""
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
        index_md_txt(rag, Path(folder), max_seq_length, characters_limit, depth, extensions)
        

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # If no arguments provided, show help
        sys.argv.append("--help")
    app(prog_name="index-markdown")
"""
