from fastapi import UploadFile
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

    def _process_single_paper(self, f: Path, rag: MeiliRAG, max_seq_length: int, characters_for_abstract: int, keep_memory: bool = False) -> List[dict]:
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
                            remember_query=not keep_memory)
                    
                    paper = Annotation.model_validate(response)
                    annotation_task.log(message_type="process_paper.annotation_complete", title=paper.title)
                except Exception as e:
                    annotation_task.log(message_type="process_paper.annotation_error", 
                                       error=str(e), 
                                       error_type=str(type(e).__name__), query=query)
                    # Re-raise the exception to maintain original behavior
                    raise
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
        
    def delete_by_source(self, index_name: str, source: str) -> None:
        """Delete documents by their sources from the MeiliRAG index.
        
        Args:
            rag: MeiliRAG instance
            sources: List of source strings to delete
        """
        rag = MeiliRAG.get_instance(
            index_name=index_name,
            model=model,        # The embedding model used for the search
        )
        rag.delete_by_source(source)
        


    def index_markdown(self, folder: Path, index_name: str) -> List[dict]:
        model_str = os.getenv("EMBEDDING_MODEL", EmbeddingModel.JINA_EMBEDDINGS_V3.value)
        model = EmbeddingModel(model_str)

        max_seq_length: Optional[int] = os.getenv("INDEX_MAX_SEQ_LENGTH", 3600)
        characters_for_abstract: int = os.getenv("INDEX_CHARACTERS_FOR_ABSTRACT", 5000)
        
        # Create and return RAG instance with conditional recreate_index
        # It should use default environment variables for host, port, api_key, create_index_if_not_exists, recreate_index
        rag = MeiliRAG.get_instance(
            index_name=index_name,
            model=model,        # The embedding model used for the search
        )
        return self.index_md_txt(rag, folder, max_seq_length, characters_for_abstract)
    

    def _create_rag_instance(self, index_name: str, model: Optional[EmbeddingModel] = None) -> MeiliRAG:
        """Create a MeiliRAG instance with default parameters.
        
        Args:
            index_name: Name of the index to create or use
            model: Optional embedding model to use (defaults to environment variable)
            
        Returns:
            MeiliRAG instance
        """
        model_str = os.getenv("EMBEDDING_MODEL", EmbeddingModel.JINA_EMBEDDINGS_V3.value)
        actual_model = model or EmbeddingModel(model_str)
        
        return MeiliRAG.get_instance(
            index_name=index_name,
            model=actual_model,
        )

    def _process_metadata(self, text_content: str, filename: str, 
                        title: Optional[str], abstract: Optional[str], source: Optional[str],
                        autoannotate: bool, characters_for_abstract: int,
                        action_log: callable) -> tuple:
        """Process document metadata with optional auto-annotation.
        
        Args:
            text_content: Document text content
            filename: Original filename
            title: Optional title for the document
            abstract: Optional abstract for the document
            source: Optional source attribution for the document
            autoannotate: Whether to use AI to extract metadata
            characters_for_abstract: Number of characters to use for annotation
            action_log: Logging function for the current task
            
        Returns:
            tuple: (title, abstract, source) - processed metadata
        """
        # Check if all metadata is missing and autoannotate is False
        if not title and not abstract and not source and not autoannotate:
            warning_msg = "Warning: All metadata (title, abstract, source) is missing and autoannotate is disabled. Using default values."
            action_log(message_type="metadata_warning", warning=warning_msg)
        
        # Process metadata
        if not title or not abstract or not source:
            if autoannotate:
                action_log(message_type="auto_annotating_document")
                # Only use part of the text for annotation
                text_sample = text_content[:characters_for_abstract]
                query = f"Extract the abstract, authors and title of the following document (from file {filename}):\n{text_sample}"
                
                try:
                    enforce_validation = os.environ.get("INDEXING_ENFORCE_VALIDATION", "False").lower() in ("true", "1", "yes")
                    response = self.annotation_agent.query_structural(
                        query, 
                        Annotation, 
                        enforce_validation=enforce_validation)
                    
                    paper = Annotation.model_validate(response)
                    
                    if not title:
                        title = paper.title
                    if not abstract:
                        abstract = paper.abstract
                    if not source:
                        source = paper.source or filename
                        
                    action_log(message_type="auto_annotation_complete", title=title)
                except Exception as e:
                    action_log(message_type="auto_annotation_error", 
                            error=str(e), 
                            error_type=str(type(e).__name__))
                    # If annotation fails, use defaults for missing values
                    if not title:
                        title = filename
                    if not abstract:
                        abstract = text_content[:200] + "..."
                    if not source:
                        source = filename
            else:
                # Use defaults if not auto-annotating
                if not title:
                    title = filename
                if not abstract:
                    abstract = text_content[:200] + "..."
                if not source:
                    source = filename
        
        return title, abstract, source

    def _process_and_index_document(self, text_content: str, title: str, abstract: str, source: str,
                                 rag: MeiliRAG, max_seq_length: int, action_log: callable) -> List[dict]:
        """Process and index a document.
        
        Args:
            text_content: Document text content
            title: Document title
            abstract: Document abstract
            source: Document source
            rag: MeiliRAG instance for document storage
            max_seq_length: Maximum sequence length for chunks
            action_log: Logging function for the current task
            
        Returns:
            List of document chunks created and indexed
        """
        # Process the document
        with start_task(message_type="splitting_document") as splitting_task:
            splitter_instance = ArticleSplitter(model=rag.sentence_transformer, max_seq_length=max_seq_length)
            authors = []  # Can be empty for single uploaded files
            docs = splitter_instance.split(text_content, title=title, abstract=abstract, authors=authors, source=source)
            splitting_task.log(message_type="splitting_complete", chunks_count=len(docs))
        
        # Add to index
        with start_task(message_type="embedding_and_indexing") as embedding_task:
            rag.add_documents(docs)
            embedding_task.log(message_type="embedding_complete", chunks_count=len(docs))
        
        return docs

    def index_pdf_file(self, file: UploadFile, index_name: str, max_seq_length: Optional[int] = 3600, 
                        abstract: Optional[str] = None, title: Optional[str] = None, source: Optional[str] = None,
                        autoannotate: bool = True, mistral_api_key: Optional[str] = None) -> str:
        """
        Accepts a PDF file upload and indexes it into the search database.
        
        Args:
            file: The uploaded PDF file (FastAPI UploadFile) - the document to be parsed and indexed
            index_name: Name of the index to create or update - determines where documents are stored
            max_seq_length: Maximum sequence length for chunks - controls how documents are split
                            (defaults to 3600 characters if None)
            abstract: Optional abstract for the document - a summary of the content
                     (defaults to first 200 chars if not provided and not auto-annotated)
            title: Optional title for the document - used for identification and reference
                  (defaults to filename if not provided and not auto-annotated)
            source: Optional source attribution for the document - indicates origin or reference
                   (defaults to filename if not provided and not auto-annotated)
            autoannotate: Whether to auto-annotate the document if metadata is missing - 
                         when True, uses AI to extract metadata from the document content
                         (defaults to False)
            mistral_api_key: Optional API key for Mistral OCR service - required for PDF parsing
                            (defaults to environment variable MISTRAL_API_KEY if not provided)
            
        Returns:
            str: Message describing the indexing results
        """
        import tempfile
        from mistral_ocr import MistralOCRParser
        
        with start_task(action_type="rag_server_index_pdf_file", index_name=index_name) as action:
            try:
                # Get API key from parameter or environment variable
                api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
                if not api_key:
                    return "Error: Mistral API key is required for PDF processing. Please provide it as a parameter or set the MISTRAL_API_KEY environment variable."
                
                # Save the uploaded PDF to a temporary file
                filename = file.filename or "uploaded_file.pdf"
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir_path = Path(temp_dir)
                    pdf_path = temp_dir_path / filename
                    markdown_path = temp_dir_path / f"{filename}.md"
                    
                    # Save the PDF file
                    content = file.file.read()
                    with open(pdf_path, "wb") as pdf_file:
                        pdf_file.write(content)
                    
                    action.log(message_type="pdf_saved", pdf_path=str(pdf_path))
                    
                    # Initialize the OCR parser with the provided API key
                    parser = MistralOCRParser(api_key=api_key)
                    
                    # Parse the PDF file to markdown
                    with start_task(message_type="parsing_pdf") as parsing_task:
                        result = parser.parse_pdf(str(pdf_path), str(markdown_path))
                        parsing_task.log(message_type="pdf_parsed", markdown_path=str(markdown_path))
                    
                    # Read the markdown content
                    text_content = markdown_path.read_text()
                    
                    # Configure parameters
                    if max_seq_length is None:
                        max_seq_length = int(os.getenv("INDEX_MAX_SEQ_LENGTH", 3600))
                    
                    characters_for_abstract = int(os.getenv("INDEX_CHARACTERS_FOR_ABSTRACT", 10000))
                    
                    # Create RAG instance
                    rag = self._create_rag_instance(index_name)
                    
                    # Process metadata
                    title, abstract, source = self._process_metadata(
                        text_content, filename, title, abstract, source, 
                        autoannotate, characters_for_abstract, action.log
                    )
                    
                    # Process and index the document
                    docs = self._process_and_index_document(
                        text_content, title, abstract, source, 
                        rag, max_seq_length, action.log
                    )
                    
                    return f"Successfully indexed PDF document '{title}' with {len(docs)} chunks into index '{index_name}'"
                    
            except Exception as e:
                error_msg = f"Error processing PDF file: {str(e)}"
                action.log(message_type="error", error=error_msg, error_type=str(type(e).__name__))
                return error_msg
    
    
    def index_text_file(self, file: UploadFile, index_name: str, max_seq_length: Optional[int] = 3600, 
                        abstract: Optional[str] = None, title: Optional[str] = None, source: Optional[str] = None,
                        autoannotate: bool = False) -> str:
        """
        Accepts a text file upload and indexes it into the search database.
        
        Args:
            file: The uploaded text file (FastAPI UploadFile) - the document content to be indexed
            index_name: Name of the index to create or update - determines where documents are stored
            max_seq_length: Maximum sequence length for chunks - controls how documents are split
                            (defaults to 3600 characters if None, most of embeddings can do up to 8K)
            abstract: Optional abstract for the document - a summary of the content
                     (defaults to first 200 chars if not provided)
            title: Optional title for the document - used for identification and reference
                  (defaults to filename if not provided)
            source: Optional source attribution for the document - indicates origin or reference
                   (defaults to filename if not provided)
            autoannotate: Whether to auto-annotate the document if metadata is missing - 
                         when True, uses AI to extract metadata from the document content
                         (defaults to False)
            
        Returns:
            str: Message describing the indexing results
        """
        with start_task(action_type="rag_server_index_text_file", index_name=index_name) as action:
            try:
                # Read file content
                content = file.file.read()
                text_content = content.decode('utf-8')
                filename = file.filename or "uploaded_file.txt"
                
                # Configure parameters
                if max_seq_length is None:
                    max_seq_length = int(os.getenv("INDEX_MAX_SEQ_LENGTH", 3600))
                
                characters_for_abstract = int(os.getenv("INDEX_CHARACTERS_FOR_ABSTRACT", 10000))
                
                # Create RAG instance
                rag = self._create_rag_instance(index_name)
                
                # Process metadata
                title, abstract, source = self._process_metadata(
                    text_content, filename, title, abstract, source, 
                    autoannotate, characters_for_abstract, action.log
                )
                
                # Process and index the document
                docs = self._process_and_index_document(
                    text_content, title, abstract, source, 
                    rag, max_seq_length, action.log
                )
                
                return f"Successfully indexed document '{title}' with {len(docs)} chunks into index '{index_name}'"
                
            except Exception as e:
                error_msg = f"Error processing text file: {str(e)}"
                action.log(message_type="error", error=error_msg, error_type=str(type(e).__name__))
                return error_msg
    
    
    def index_upload_markdown_folder(self, uploaded_file: UploadFile, index_name: str) -> str:
        """
        Accepts a zip file upload, extracts it to a temporary directory,
        and indexes the markdown files within using index_markdown_folder.
        
        Args:
            uploaded_file: The uploaded zip file (FastAPI UploadFile)
            index_name: Name of the index to create or update
            
        Returns:
            str: Message describing the indexing results
        """
        import tempfile
        import zipfile
        from fastapi import UploadFile
        
        with start_task(action_type="rag_server_upload_and_index_zip", index_name=index_name) as action:
            try:
                # Create a temporary directory for extraction
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    # Save the uploaded file to temp location
                    temp_zip_path = temp_path / "uploaded.zip"
                    with open(temp_zip_path, "wb") as temp_file:
                        content = uploaded_file.file.read()
                        temp_file.write(content)
                    
                    # Extract the zip file
                    extraction_path = temp_path / "extracted"
                    extraction_path.mkdir(exist_ok=True)
                    
                    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extraction_path)
                    
                    action.log(message_type="zip_extracted", extraction_path=str(extraction_path))
                    
                    # Index the extracted folder
                    result = self.index_markdown_folder(extraction_path, index_name)
                    action.log(message_type="indexing_complete")
                    
                    return result
                    
            except zipfile.BadZipFile:
                error_msg = "The uploaded file is not a valid zip archive"
                action.log(message_type="error", error=error_msg)
                return error_msg
            except Exception as e:
                error_msg = f"Error processing zip file: {str(e)}"
                action.log(message_type="error", error=error_msg, error_type=str(type(e).__name__))
                return error_msg

    def index_markdown_folder(self, folder: str | Path, index_name: str) -> str:
        """
        Indexes a folder with markdown files. The server should have access to the folder.
        Uses defensive checks for documents that might be either dicts or Document instances.
        Reports errors to Eliot logs without breaking execution; problematic documents are skipped.
        """
        
        with start_task(action_type="rag_server_index_markdown_folder", folder=folder, index_name=index_name) as action:
            folder_path = Path(folder) if isinstance(folder, str) else folder
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
                rag = MeiliRAG.get_instance(
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
        rag = MeiliRAG.get_instance(
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