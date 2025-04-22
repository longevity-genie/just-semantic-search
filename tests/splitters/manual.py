from pathlib import Path
from litellm import BaseModel
import typer
from typing import Optional
import os
from dotenv import load_dotenv
from just_semantic_search.splitters.splitter_factory import SplitterType, create_splitter
from just_semantic_search.embeddings import EmbeddingModel, load_sentence_transformer_from_enum
from pycomfort.logging import to_nice_file, to_nice_stdout
from pycomfort.files import with_ext
from eliot import start_task, to_file as eliot_to_file
import json

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
    data_path: str = typer.Option("data/lifespan_json/posts_flat_blog", "--data-path", "-d", help="Path to data directory or file"),
    splitter_type: SplitterType = typer.Option(SplitterType.TEXT.value, "--splitter-type", "-s", help="Type of splitter to use"),
    model_name: EmbeddingModel = typer.Option(EmbeddingModel.JINA_EMBEDDINGS_V3.value, "--model", "-m", help="Embedding model to use"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size for processing"),
    similarity_threshold: float = typer.Option(0.8, "--similarity-threshold", "-t", help="Threshold for semantic similarity"),
    min_token_count: int = typer.Option(500, "--min-token-count", "-c", help="Minimum token count for chunks"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for saving results"),
    normalize_embeddings: bool = typer.Option(True, "--normalize", "-n", help="Whether to normalize embeddings"),
):
    """
    Run a specified splitter on documents in the given path and optionally save the results.
    """
    data_path = Path(data_path)
    
    with start_task(action_type="run_splitter", 
                   data_path=str(data_path), 
                   splitter_type=splitter_type.name,
                   model_name=model_name.name,
                   batch_size=batch_size) as task:
        
        test = with_ext(data_path, "json").first()
        # Parse the JSON content instead of just printing it
        json_content = json.loads(test.read_text())
        print(json_content)
        """
        task.log(message_type="loading_model")
        model = load_sentence_transformer_from_enum(model_name)
        task.log(message_type="model_loaded")
        
        
        # Create the splitter
        splitter = create_splitter(
            splitter_type=splitter_type,
            model=model,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            similarity_threshold=similarity_threshold,
            min_token_count=min_token_count
        )
        
        #task.log(message_type="splitter_created", splitter_class=splitter.__class__.__name__)
        
        # Process files
        result_documents = []
        
        if data_path.is_file():
            task.log(message_type="processing_file", file=str(data_path))
            documents = splitter.split_file(data_path)
            result_documents.extend(documents)
            task.log(message_type="file_processed", document_count=len(documents))
        elif data_path.is_dir():
            task.log(message_type="processing_directory", directory=str(data_path))
            documents = splitter.split_folder(data_path)
            result_documents.extend(documents)
            task.log(message_type="directory_processed", document_count=len(documents))
        else:
            typer.echo(f"Error: {data_path} does not exist or is not accessible.")
            return
        
        # Print summary
        typer.echo(f"Processed {len(result_documents)} documents using {splitter_type.name} splitter")
        
        # Save results if output file is specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
            task.log(message_type="saving_results", output_file=str(output_path))
            
            # Convert documents to JSON-serializable format
            documents_json = [doc.model_dump() for doc in result_documents]
            
            with open(output_path, 'w') as f:
                json.dump(documents_json, f, indent=2)
            
            task.log(message_type="results_saved")
            typer.echo(f"Results saved to {output_path}")
        """
        


if __name__ == "__main__":
    app()

