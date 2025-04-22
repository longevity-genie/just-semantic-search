from eliot import start_task
from fastapi import UploadFile
from just_agents.base_agent import BaseAgent
from just_semantic_search.splitters.article_splitter import ArticleSplitter
from just_semantic_search.embeddings import EmbeddingModel
from just_semantic_search.meili.rag import Document, EmbeddingModel, List, MeiliRAG, Path, os
from just_semantic_search.server.indexing import Annotation, Indexing
from just_semantic_search.splitters.text_splitters import Document, List, Path
from pycomfort import files


import os
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple, Callable


class OCRMixin:
    """Mixin providing OCR capabilities for document indexing."""

    def _process_pdf_with_ocr(self, pdf_path: Path, markdown_path: Path, api_key: str) -> str:
        """Process a PDF file with OCR and convert it to markdown.
        
        Args:
            pdf_path: Path to the PDF file
            markdown_path: Path where the markdown output should be saved
            api_key: API key for the OCR service
            
        Returns:
            str: The extracted text content
        """
        from mistral_ocr import MistralOCRParser
        
        with start_task(message_type="parsing_pdf") as parsing_task:
            # No need to wrap log function with SkipValidation
            # Initialize the OCR parser with the provided API key
            parser = MistralOCRParser(api_key=api_key)
            
            # Parse the PDF file to markdown
            result = parser.parse_pdf(str(pdf_path), str(markdown_path))
            parsing_task.log(message_type="pdf_parsed", markdown_path=str(markdown_path))
            
            # Read the markdown content
            text_content = markdown_path.read_text()
            return text_content


class AgenticIndexing(Indexing, OCRMixin):

    annotation_agent: BaseAgent

    def _process_single_paper(self, f: Path, rag: MeiliRAG, max_seq_length: int, characters_for_abstract: int, keep_memory: bool = False) -> List[dict]:
        """Process a single paper file and add it to the RAG index.

        Args:
            f: Path to the file
            rag: MeiliRAG instance for document storage
            max_seq_length: Maximum sequence length for chunks
            characters_for_abstract: Number of characters to use for extracting metadata
            keep_memory: Whether to keep memory of the query

        Returns:
            List of document chunks created from this paper
        """
        with start_task(message_type="process_paper", file=str(f.name)) as file_task:
            text = f.read_text()
            
            # No need to wrap log function with SkipValidation
            # Process metadata using parent method
            title, abstract, source = self._process_metadata(
                text_content=text,
                filename=f.name,
                title=None,
                abstract=None,
                source=None,
                autoannotate=True,
                characters_for_abstract=characters_for_abstract,
                action_log=file_task.log
            )
            
            # Process and index the document using parent method
            docs = self._process_and_index_document(
                text_content=text,
                title=title,
                abstract=abstract,
                source=source,
                rag=rag,
                max_seq_length=max_seq_length,
                action_log=file_task.log
            )
            
            file_task.log(message_type="process_paper.indexed", document_count=len(docs))
            return docs
        
    
    def annotate_metadata(self, text_content: str, filename: str,
                        title: Optional[str], abstract: Optional[str], source: Optional[str],
                        autoannotate: bool, characters_for_abstract: int,
                        action_log: Callable) -> Annotation:
        """Process document metadata with agent annotation.

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
            Annotation: The extracted metadata as an Annotation object
        """
        if not autoannotate:
            return super().annotate_metadata(text_content, filename, title, abstract, source, autoannotate, characters_for_abstract, action_log)
        else:   
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
                action_log(message_type="auto_annotation_complete", title=paper.title)
                return paper
            except Exception as e:
                action_log(message_type="auto_annotation_error",
                        error=str(e),
                        error_type=str(type(e).__name__))
                # If annotation fails, return default values
                return Annotation(
                    title=title or filename,
                    abstract=abstract or text_content[:200] + "...",
                    source=source or filename
                )


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
            
            # No need to wrap task.log with SkipValidation

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

    def index_pdf_file(self, file: UploadFile, index_name: str, max_seq_length: Optional[int] = 3600,
                        abstract: Optional[str] = None, title: Optional[str] = None, source: Optional[str] = None,
                        autoannotate: bool = True, mistral_api_key: Optional[str] = None, api_key: Optional[str] = None) -> str:
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
            api_key: Optional API key for authentication - used to secure the endpoint
                    (defaults to environment variable INDEXING_API_KEY if not provided)

        Returns:
            str: Message describing the indexing results
        """
        with start_task(action_type="rag_server_index_pdf_file", index_name=index_name) as action:
            try:
                # Verify API key if it's set in the environment
                env_api_key = os.getenv("INDEXING_API_KEY")
                if env_api_key and (not api_key or api_key != env_api_key):
                    return "Error: Invalid or missing API key for indexing operations"

                # Get API key from parameter or environment variable
                ocr_api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
                if not ocr_api_key:
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

                    # Process PDF with OCR
                    text_content = self._process_pdf_with_ocr(pdf_path, markdown_path, ocr_api_key)

                    # Configure parameters
                    if max_seq_length is None:
                        max_seq_length = int(os.getenv("INDEX_MAX_SEQ_LENGTH", 3600))

                    characters_for_abstract = int(os.getenv("INDEX_CHARACTERS_FOR_ABSTRACT", 10000))

                    # Create RAG instance
                    rag = self._create_rag_instance(index_name)
                    
                    # No need to wrap the log function with SkipValidation
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
