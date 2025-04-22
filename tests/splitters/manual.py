from pathlib import Path
from just_semantic_search.splitters.structural_splitters import DictionarySplitter
from sentence_transformers import SentenceTransformer
import typer
from typing import Optional, List
from dotenv import load_dotenv
from just_semantic_search.splitters.splitter_factory import SplitterType, create_splitter
from just_semantic_search.embeddings import EmbeddingModel, load_sentence_transformer_from_enum
from pycomfort.logging import to_nice_file, to_nice_stdout
from eliot import start_task
import json
from rich.pretty import pprint
import os
from just_semantic_search.meili.rag import MeiliRAG
from just_semantic_search.meili.utils.services import ensure_meili_is_running

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
json_log = log_dir / "splitter_run.json"
rendered_log = log_dir / "splitter_run.txt"
to_nice_file(json_log, rendered_file=rendered_log)
to_nice_stdout()

# Load environment variables
load_dotenv(override=True)

app = typer.Typer()


@app.command()
def run_splitter(
    json_path: Path = typer.Option("data/lifespan_json/11th_ARDD_Meeting_Announces_Speaker_Lineup.json", "--json-path", "-j", help="Path to json file"),
    #data_path: str = typer.Option("data/lifespan_json/posts_flat_blog", "--data-path", "-d", help="Path to data directory or file"),
    splitter_type: SplitterType = typer.Option(SplitterType.FLAT_JSON.value, "--splitter-type", "-s", help="Type of splitter to use"),
    model_name: EmbeddingModel = typer.Option(EmbeddingModel.JINA_EMBEDDINGS_V3.value, "--model", "-m", help="Embedding model to use"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size for processing"),
    similarity_threshold: float = typer.Option(0.8, "--similarity-threshold", "-t", help="Threshold for semantic similarity"),
    min_token_count: int = typer.Option(500, "--min-token-count", "-c", help="Minimum token count for chunks"),
    output_folder: Path = typer.Option("data/lifespan_json/output", "--output", "-o", help="Output folder for saving split files"),
    normalize_embeddings: bool = typer.Option(True, "--normalize", "-n", help="Whether to normalize embeddings"),
):
    """
    Run a specified splitter on documents in the given path and optionally save the results.
    """
    
    with start_task(action_type="run_splitter", 
                   data_path=str(json_path), 
                   splitter_type=splitter_type.name,
                   model_name=model_name.name,
                   batch_size=batch_size) as task:
        
        # Parse the JSON content instead of just printing it
        model: SentenceTransformer = load_sentence_transformer_from_enum(model_name)
        task.log(message_type="model_loaded")
        
        
        # Create the splitter
        splitter: DictionarySplitter = create_splitter(
            splitter_type=splitter_type,
            model=model,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            similarity_threshold=similarity_threshold,
            min_token_count=min_token_count,
            max_seq_length=512
        )
        

        docs = splitter.split_file(json_path)
        
        # Save results if output folder is specified
        if output_folder:
            output_path = Path(output_folder)
            output_path.mkdir(exist_ok=True, parents=True)
            
            task.log(message_type="saving_results", output_folder=str(output_path))
            
            # Save each document as a separate file
            total_docs = len(docs)
            for i, doc in enumerate(docs):
                if total_docs == 1:
                    file_path = output_path / "doc.json"
                else:
                    file_path = output_path / f"doc_{i+1}_of_{total_docs}.json"
                with open(file_path, 'w') as f:
                    json.dump(doc.model_dump(), f, indent=2)
            
            task.log(message_type="results_saved")
            typer.echo(f"Results saved to {output_path}")
        
        


@app.command()
def run_splitter_folder(
    folder_path: Path = typer.Option("data/lifespan_json/posts_flat_blog", "--folder-path", "-f", help="Path to folder containing JSON files"),
    pattern: str = typer.Option("*.json", "--pattern", "-p", help="Pattern to match JSON files"),
    splitter_type: SplitterType = typer.Option(SplitterType.FLAT_JSON.value, "--splitter-type", "-s", help="Type of splitter to use"),
    model_name: EmbeddingModel = typer.Option(EmbeddingModel.JINA_EMBEDDINGS_V3.value, "--model", "-m", help="Embedding model to use"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size for processing"),
    similarity_threshold: float = typer.Option(0.8, "--similarity-threshold", "-t", help="Threshold for semantic similarity"),
    min_token_count: int = typer.Option(500, "--min-token-count", "-c", help="Minimum token count for chunks"),
    output_folder: Path = typer.Option("data/lifespan_json/output", "--output", "-o", help="Output folder for saving split files"),
    normalize_embeddings: bool = typer.Option(True, "--normalize", "-n", help="Whether to normalize embeddings"),
):
    """
    Run a specified splitter on all JSON files in the given folder and save the results.
    """
    
    with start_task(action_type="run_splitter_folder", 
                   folder_path=str(folder_path), 
                   pattern=pattern,
                   splitter_type=splitter_type.name,
                   model_name=model_name.name,
                   batch_size=batch_size) as task:
        
        # Load model
        model: SentenceTransformer = load_sentence_transformer_from_enum(model_name)
        task.log(message_type="model_loaded")
        
        # Create the splitter
        splitter: DictionarySplitter = create_splitter(
            splitter_type=splitter_type,
            model=model,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            similarity_threshold=similarity_threshold,
            min_token_count=min_token_count
        )
        
        # Get all JSON files in the folder
        folder_path = Path(folder_path)
        json_files = list(folder_path.glob(pattern))
        
        if not json_files:
            typer.echo(f"No JSON files found in {folder_path} matching pattern '{pattern}'")
            return
        
        task.log(message_type="processing_files", file_count=len(json_files))
        typer.echo(f"Processing {len(json_files)} JSON files...")
        
        # Create output folder if it doesn't exist
        if output_folder:
            output_path = Path(output_folder)
            output_path.mkdir(exist_ok=True, parents=True)
        
        # Process each file
        for file_idx, json_path in enumerate(json_files):
            try:
                typer.echo(f"Processing file {file_idx+1}/{len(json_files)}: {json_path.name}")
                
                # Process the file
                docs = splitter.split_file(json_path)
                
                # Save results
                if output_folder:
                    total_docs = len(docs)
                    for i, doc in enumerate(docs):
                        if total_docs == 1:
                            file_path = output_path / f"{json_path.stem}.json"
                        else:
                            file_path = output_path / f"{json_path.stem}_{i+1}_of_{total_docs}.json"
                        with open(file_path, 'w') as f:
                            json.dump(doc.model_dump(), f, indent=2)
                    
                    task.log(message_type="file_processed", file_name=json_path.name)
            except Exception as e:
                task.log(message_type="error_processing_file", file_name=json_path.name, error=str(e))
                typer.echo(f"Error processing {json_path.name}: {str(e)}")
        
        task.log(message_type="all_files_processed")
        typer.echo(f"All files processed. Results saved to {output_folder}")


@app.command()
def index_json_folder(
    folder_path: Path = typer.Option("data/lifespan_json", "--folder-path", "-f", help="Path to folder containing JSON files"),
    index_name: str = typer.Option("lifespan", "--index-name", "-n", help="Name of the Meilisearch index"),
    model_name: EmbeddingModel = typer.Option(EmbeddingModel.JINA_EMBEDDINGS_V3.value, "--model", "-m", help="Embedding model to use"),
    host: str = typer.Option("127.0.0.1", "--host", help="Meilisearch host"),
    port: int = typer.Option(7700, "--port", help="Meilisearch port"),
    api_key: str = typer.Option(None, "--api-key", "-k", help="Meilisearch API key"),
    ensure_server: bool = typer.Option(True, "--ensure-server", "-e", help="Ensure Meilisearch server is running"),
    recreate_index: bool = typer.Option(False, "--recreate", "-r", help="Whether to recreate the index"),
    meili_service_dir: Path = typer.Option(Path("services/meilisearch"), "--meili-dir", help="Path to Meilisearch service directory"),
):
    """
    Index a folder of JSON files using MeiliRAG with FlatJSON splitter.
    """
    
    if api_key is None:
        api_key = os.getenv("MEILI_MASTER_KEY", "fancy_master_key")
        
    with start_task(action_type="index_json_folder", 
                   folder_path=str(folder_path),
                   index_name=index_name,
                   model_name=model_name.name,
                   host=host,
                   port=port) as task:
        
        if ensure_server:
            task.log(message_type="ensuring_server", host=host, port=port)
            ensure_meili_is_running(meili_service_dir, host, port)
            
        # Create MeiliRAG instance
        rag = MeiliRAG(
            index_name=index_name,
            model=model_name,
            host=host,
            port=port,
            api_key=api_key,
            create_index_if_not_exists=True,
            recreate_index=recreate_index
        )
        
        task.log(message_type="indexing_folder", folder=str(folder_path))
        typer.echo(f"Indexing JSON files in {folder_path} using FlatJSON splitter...")
        
        # Index the folder using MeiliRAG's index_folder method with FLAT_JSON splitter
        try:
            result = rag.index_folder(
                folder=folder_path,
                splitter=SplitterType.FLAT_JSON,
                filter=lambda x: x.suffix == ".json"
            )
            
            task.log(message_type="indexing_complete", documents_added=len(result) if result else 0)
            typer.echo(f"Indexing complete. Added documents to index '{index_name}'")
            
        except Exception as e:
            task.log(message_type="indexing_error", error=str(e))
            typer.echo(f"Error indexing files: {str(e)}")


if __name__ == "__main__":
    app()

