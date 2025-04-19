from fastapi import UploadFile
from just_semantic_search.article_splitter import ArticleSplitter
from typing import List, Optional
from just_semantic_search.meili.utils.services import ensure_meili_is_running
from just_semantic_search.server.agentic_indexing import AgenticIndexing
from just_semantic_search.server.utils import default_annotation_agent, get_project_directories, load_environment_files
from pydantic import BaseModel, Field
from just_semantic_search.text_splitters import *
from just_semantic_search.embeddings import *

from just_semantic_search.utils.tokens import *
from pathlib import Path
from just_agents import llm_options

import typer
import os
from just_semantic_search.meili.rag import *
from pathlib import Path

from eliot._output import *
from eliot import start_task


from pathlib import Path
from pycomfort import files
from eliot import start_task


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
    
    
    def delete_by_source(self, index_name: str, source: str, api_key: Optional[str] = None) -> str:
        """Delete documents by their sources from the MeiliRAG index.
        
        Args:
            index_name: Name of the index to delete from
            source: Source string to delete
            api_key: Optional API key for authentication - used to secure the endpoint
                    (defaults to environment variable INDEXING_API_KEY if not provided)
            
        Returns:
            str: Message describing the deletion results
        """
        with start_task(action_type="rag_server_delete_by_source", index_name=index_name, source=source) as action:
            try:
                # Verify API key if it's set in the environment
                env_api_key = os.getenv("INDEXING_API_KEY")
                if env_api_key and (not api_key or api_key != env_api_key):
                    return "Error: Invalid or missing API key for indexing operations"
                
                rag = MeiliRAG.get_instance(
                    index_name=index_name
                )
                
                deleted_count = rag.delete_by_source(source)
                
                return f"Successfully deleted {deleted_count} documents with source '{source}' from index '{index_name}'"
            except Exception as e:
                error_msg = f"Error deleting documents: {str(e)}"
                action.log(message_type="error", error=error_msg, error_type=str(type(e).__name__))
                return error_msg

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
    
    def annotate_metadata(self, text_content: str, filename: str,
                        title: Optional[str], abstract: Optional[str], source: Optional[str],
                        autoannotate: bool, characters_for_abstract: int,
                        action_log: callable) -> Annotation:
        """Process document metadata with optional auto-annotation.
        Args:
            text_content: Document text content
        """
        if autoannotate == False:
            return Annotation(
                title=title or filename,
                abstract=abstract or text_content[:200] + "...",
                source=source or filename
            )
        else:
            raise NotImplementedError("annotate_metadata is not implemented")

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
            annotation = self.annotate_metadata(
                text_content=text_content,
                filename=filename,
                title=title,
                abstract=abstract,
                source=source,
                autoannotate=autoannotate,
                characters_for_abstract=characters_for_abstract,
                action_log=action_log
            )
            
            # Use annotation results for any missing metadata
            if not title:
                title = annotation.title
            if not abstract:
                abstract = annotation.abstract
            if not source:
                source = annotation.source or filename

        return title, abstract, source
    
    def _process_and_index_document(self, text_content: str, title: str, abstract: str, source: str,
                                  rag: MeiliRAG, max_seq_length: int, action_log: callable) -> List[dict]:
        """Process and index a document.
        
        Args:
            text_content: The text content to process
            title: Title for the document
            abstract: Abstract for the document
            source: Source attribution for the document
            rag: MeiliRAG instance for indexing
            max_seq_length: Maximum sequence length for chunks
            action_log: Logging function for the current task
            
        Returns:
            List[dict]: The indexed document chunks
        """
        with start_task(message_type="processing_document") as processing_task:
            # Create document splitter
            splitter_instance = ArticleSplitter(model=rag.sentence_transformer, max_seq_length=max_seq_length)
            
            # Split the document
            docs = splitter_instance.split(text_content, title=title, abstract=abstract, source=source)
            processing_task.log(message_type="document_split", chunks_count=len(docs))
            
            # Add documents to RAG
            rag.add_documents(docs)
            processing_task.log(message_type="document_indexed", chunks_count=len(docs))
            
            return docs

    def index_text_file(self, file: UploadFile, index_name: str, max_seq_length: Optional[int] = 3600,
                        abstract: Optional[str] = None, title: Optional[str] = None, source: Optional[str] = None,
                        autoannotate: bool = False, api_key: Optional[str] = None) -> str:
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
            api_key: Optional API key for authentication - used to secure the endpoint
                    (defaults to environment variable INDEXING_API_KEY if not provided)

        Returns:
            str: Message describing the indexing results
        """
        with start_task(action_type="rag_server_index_text_file", index_name=index_name) as action:
            try:
                # Verify API key if it's set in the environment
                env_api_key = os.getenv("INDEXING_API_KEY")
                if env_api_key and (not api_key or api_key != env_api_key):
                    return "Error: Invalid or missing API key for indexing operations"

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
            
    def index_upload_markdown_folder(self, uploaded_file: UploadFile, index_name: str, api_key: Optional[str] = None) -> str:
        """
        Accepts a zip file upload, extracts it to a temporary directory,
        and indexes the markdown files within using index_markdown_folder.

        Args:
            uploaded_file: The uploaded zip file (FastAPI UploadFile)
            index_name: Name of the index to create or update
            api_key: Optional API key for authentication - used to secure the endpoint
                    (defaults to environment variable INDEXING_API_KEY if not provided)

        Returns:
            str: Message describing the indexing results
        """
        import tempfile
        import zipfile
        from fastapi import UploadFile

        with start_task(action_type="rag_server_upload_and_index_zip", index_name=index_name) as action:
            try:
                # Verify API key if it's set in the environment
                env_api_key = os.getenv("INDEXING_API_KEY")
                if env_api_key and (not api_key or api_key != env_api_key):
                    return "Error: Invalid or missing API key for indexing operations"

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
                    result = self.index_markdown_folder(extraction_path, index_name, api_key=api_key)
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
            
    def index_markdown_folder(self, folder: str | Path, index_name: str, api_key: Optional[str] = None) -> str:
        """
        Indexes a folder with markdown files. The server should have access to the folder.
        Uses defensive checks for documents that might be either dicts or Document instances.
        Reports errors to Eliot logs without breaking execution; problematic documents are skipped.

        Args:
            folder: Path to the folder containing markdown files
            index_name: Name of the index to create or update
            api_key: Optional API key for authentication - used to secure the endpoint
                    (defaults to environment variable INDEXING_API_KEY if not provided)
        """

        with start_task(action_type="rag_server_index_markdown_folder", folder=folder, index_name=index_name) as action:
            # Verify API key if it's set in the environment
            env_api_key = os.getenv("INDEXING_API_KEY")
            if env_api_key and (not api_key or api_key != env_api_key):
                return "Error: Invalid or missing API key for indexing operations"

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
    
    def index_json_files(self, folder: str | Path, index_name: str, 
                         content_field: str,  
                         max_seq_length: Optional[int] = None,
                         api_key: Optional[str] = None,
                         extension: str = ".json",
                         depth: int = -1,
                         required_fields: Optional[List[str]] = None) -> str:
        """
        Indexes a folder with JSON files with fully custom fields. Each JSON file should be a document 
        or contain an array of documents. All fields from the JSON will be preserved in the index.
        
        Args:
            folder: Path to the folder containing JSON files
            index_name: Name of the index to create or update
            content_field: Field name in the JSON that contains the document content (required for splitting)
            max_seq_length: Maximum sequence length for chunks
            api_key: Optional API key for authentication
            extension: File extension to look for (default: .json)
            depth: Depth of folder traversal (-1 for unlimited)
            required_fields: Optional list of field names that must be present in each document
                    
        Returns:
            str: Message describing the indexing results
        """
        import json
        from copy import deepcopy
        
        with start_task(action_type="rag_server_index_json_files", folder=folder, index_name=index_name) as action:
            try:
                # Verify API key if it's set in the environment
                env_api_key = os.getenv("INDEXING_API_KEY")
                if env_api_key and (not api_key or api_key != env_api_key):
                    return "Error: Invalid or missing API key for indexing operations"
                
                folder_path = Path(folder) if isinstance(folder, str) else folder
                if not folder_path.exists():
                    msg = f"Folder {folder} does not exist or the server does not have access to it"
                    action.log(msg)
                    return msg
                
                with start_task(action_type="rag_server_index_json_files.config") as config_task:
                    model_str = os.getenv("EMBEDDING_MODEL", EmbeddingModel.JINA_EMBEDDINGS_V3.value)
                    model = EmbeddingModel(model_str)
                    
                    if max_seq_length is None:
                        max_seq_length = int(os.getenv("INDEX_MAX_SEQ_LENGTH", 3600))
                    config_task.log(message_type="config_loaded", model=model_str, max_seq_length=max_seq_length)
                
                # Create RAG instance
                with start_task(action_type="rag_server_index_json_files.create_rag") as rag_task:
                    rag = MeiliRAG.get_instance(
                        index_name=index_name,
                        model=model,
                    )
                    rag_task.log(message_type="rag_created", index_name=index_name)
                
                # Process all JSON files in the folder
                with start_task(action_type="rag_server_index_json_files.processing") as processing_task:
                    json_files = files.traverse(folder_path, lambda x: x.suffix == extension, depth=depth)
                    processing_task.log(message_type="found_json_files", count=len(json_files))
                    
                    total_docs = 0
                    errors = 0
                    
                    for json_file in json_files:
                        try:
                            with start_task(action_type="processing_json_file", file=str(json_file)) as file_task:
                                # Load JSON content
                                with open(json_file, 'r') as f:
                                    json_content = json.load(f)
                                
                                # Check if it's a list or a single document
                                if isinstance(json_content, list):
                                    docs = json_content
                                else:
                                    docs = [json_content]
                                
                                file_task.log(message_type="json_loaded", documents=len(docs))
                                
                                # Process each document
                                for doc in docs:
                                    try:
                                        # Check for required content field
                                        if content_field not in doc:
                                            file_task.log(message_type="missing_content_field", 
                                                        field=content_field, 
                                                        document=str(doc)[:100])
                                            continue
                                        
                                        # Check for any other required fields
                                        if required_fields:
                                            missing_fields = [field for field in required_fields if field not in doc]
                                            if missing_fields:
                                                file_task.log(message_type="missing_required_fields", 
                                                            fields=missing_fields, 
                                                            document=str(doc)[:100])
                                                continue
                                        
                                        # Get the main content for splitting
                                        content = doc[content_field]
                                        
                                        # Create a copy of the document to preserve all original fields
                                        metadata = deepcopy(doc)
                                        # Create an empty list for chunks
                                        chunks = []
                                        
                                        # Split the content field
                                        splitter_instance = ArticleSplitter(model=rag.sentence_transformer, 
                                                                        max_seq_length=max_seq_length)
                                        
                                        # Here, we're using a base document with all original fields
                                        # and just splitting the content field
                                        raw_splits = splitter_instance._split_text(content)
                                        
                                        # Create a chunk for each split with all the original metadata
                                        for i, split in enumerate(raw_splits):
                                            chunk = deepcopy(metadata)
                                            # Replace the content with just this chunk
                                            chunk[content_field] = split
                                            # Add chunk metadata
                                            chunk['_chunk_id'] = i
                                            chunk['_total_chunks'] = len(raw_splits)
                                            chunk['_source_file'] = str(json_file.name)
                                            
                                            # Convert to document format expected by MeiliRAG
                                            chunks.append(chunk)
                                        
                                        # Add to index
                                        rag.add_documents(chunks)
                                        total_docs += len(chunks)
                                        
                                        file_task.log(message_type="document_processed", 
                                                    chunks=len(chunks),
                                                    fields=list(doc.keys()))
                                    except Exception as e:
                                        errors += 1
                                        file_task.log(message_type="document_processing_error", 
                                                    error=str(e), 
                                                    error_type=str(type(e).__name__),
                                                    document=str(doc)[:100])
                                        continue
                        except Exception as e:
                            errors += 1
                            processing_task.log(message_type="file_processing_error", 
                                            file=str(json_file),
                                            error=str(e), 
                                            error_type=str(type(e).__name__))
                            continue
                    
                    processing_task.log(message_type="json_processing_complete", 
                                    total_documents=total_docs,
                                    errors=errors)
                    
                    return f"Successfully indexed {total_docs} document chunks from JSON files in {folder_path}. Encountered {errors} errors."
            except Exception as e:
                error_msg = f"Error processing JSON files: {str(e)}"
                action.log(message_type="error", error=error_msg, error_type=str(type(e).__name__))
                return error_msg