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
    folder_path: Path = typer.Option("data/lifespan_json", "--folder-path", "-f", help="Path to folder containing JSON files"),
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
            min_token_count=min_token_count,
            max_seq_length=512
        )
        
        # Get all JSON files in the folder
        folder_path = Path(folder_path)
        json_files = list(folder_path.glob(pattern))
        
        if not json_files:
            typer.echo(f"No JSON files found in {folder_path} matching pattern '{pattern}'")
            return
        
        task.log(message_type="processing_files", file_count=len(json_files))
        typer.echo(f"Processing {len(json_files)} JSON files...")
        
        # Process each file
        for file_idx, json_path in enumerate(json_files):
            try:
                typer.echo(f"Processing file {file_idx+1}/{len(json_files)}: {json_path.name}")
                
                # Create a subfolder for each file's output
                if output_folder:
                    file_output_folder = Path(output_folder) / json_path.stem
                    file_output_folder.mkdir(exist_ok=True, parents=True)
                    
                    # Process the file
                    docs = splitter.split_file(json_path)
                    
                    # Save results
                    total_docs = len(docs)
                    for i, doc in enumerate(docs):
                        if total_docs == 1:
                            file_path = file_output_folder / "doc.json"
                        else:
                            file_path = file_output_folder / f"doc_{i+1}_of_{total_docs}.json"
                        with open(file_path, 'w') as f:
                            json.dump(doc.model_dump(), f, indent=2)
                    
                    task.log(message_type="file_processed", file_name=json_path.name)
            except Exception as e:
                task.log(message_type="error_processing_file", file_name=json_path.name, error=str(e))
                typer.echo(f"Error processing {json_path.name}: {str(e)}")
        
        task.log(message_type="all_files_processed")
        typer.echo(f"All files processed. Results saved to {output_folder}")


if __name__ == "__main__":
    app()

