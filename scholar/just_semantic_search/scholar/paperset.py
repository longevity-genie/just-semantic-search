from datetime import datetime
from typing import TypeVar
import typer
import polars as pl
from pathlib import Path
from typing import Optional, TypeVar
from just_semantic_search.paragraph_splitters import *
import patito as pt
from rich.pretty import pprint
from just_semantic_search.utils.logs import to_nice_file, to_nice_stdout
from just_semantic_search.utils.models import get_sentence_transformer_model_name
from sentence_transformers import SentenceTransformer
from just_semantic_search.embeddings import *
from just_semantic_search.utils.tokens import *
from pathlib import Path
from just_semantic_search.paragraph_splitters import *


import typer
import os
from just_semantic_search.meili.rag import *
from just_semantic_search.meili.rag import *
import time
from pathlib import Path

from eliot._output import *
from eliot import Action, start_task
from just_semantic_search.meili.utils.services import ensure_meili_is_running


from just_semantic_search.scholar.papers import SCHOLAR_MAIN_COLUMNS, Paper


project_dir = Path(__file__).parent.parent.parent.parent
print(f"project_dir: {project_dir}")
data_dir = project_dir / "data"
logs = project_dir / "logs"
meili_service_dir = project_dir / "services" / "meili"

default_output_dir = Path(__file__).parent.parent.parent / "data"

# Configure Eliot to output to both stdout and log files
log_file_path = logs / f"paperset_meili_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logs.mkdir(exist_ok=True)  # Ensure logs directory exists

# Create both JSON and rendered log files
json_log = open(f"{log_file_path}.json", "w")
rendered_log = open(f"{log_file_path}.txt", "w")


# Set up logging destinations
#to_file(sys.stdout)  # Keep console output
to_nice_stdout() #to stdout
to_nice_file(json_log, rendered_file=rendered_log)


T = TypeVar('T')

pl.Config.set_tbl_rows(-1)  # Show all rows
pl.Config.set_tbl_cols(-1)  # Show all columns
pl.Config.set_fmt_str_lengths(1000)  # Increase string length in output

app = typer.Typer()


@app.command()
def papers(df_name_or_path: str = "hf://datasets/longevity-genie/aging_papers_paragraphs/aging_specific_pubmed.parquet"):
    #frame: pl.LazyFrame = pl.scan_parquet(df_name_or_path)
    #df = pt.LazyFrame[Paper](frame).set_model(Paper)
    frame = pl.read_parquet(df_name_or_path)
    df = pt.DataFrame(frame).set_model(Paper)
    results = df.head()
    print("RESULTS:")
    print(results)


@app.command()
def test(df_name_or_path: str = "hf://datasets/longevity-genie/tacutu_papers/tacutu_pubmed.parquet"):
    cols = SCHOLAR_MAIN_COLUMNS
    frame = pl.read_parquet(df_name_or_path, columns=cols)
    #df = pt.DataFrame[Paper](frame).set_model(Paper)
    #for paper in df.head().iter_models():
    #    pprint(paper)
    pprint(frame.head(4))


def process_batch(papers_batch: list[Paper], splitter: ParagraphTextSplitter, rag: MeiliRAG):
            """
            Processes batch of papers to be added to the database"""
            start_time = time.time()  # Start timing
            with start_action(action_type="process batch") as action:
                batch_documents = []
                for paper in papers_batch:
                    paper_start = time.time()  # Time individual paper
                    paragraphs = paper.annotations_paragraph
                    source = paper.externalids_doi if paper.externalids_doi else paper.externalids_pubmed
                    documents = splitter.split(paragraphs, source=source, title=paper.title, abstract=paper.abstract, references=paper.references)
                    batch_documents.extend(documents)
                    token_counts = sum([d.token_count for d in documents])
                    paper_time = time.time() - paper_start
                    action.log(message_type="paper_processing_time", 
                             paper_id=source, 
                             time=paper_time)
                rag.add_documents(batch_documents)
                
                batch_time = time.time() - start_time
                action.add_success_fields(
                    batch_size=len(batch_documents),
                    batch_processing_time=batch_time,
                    avg_time_per_paper=batch_time/len(papers_batch),
                    total_tokens=token_counts
                )
            return batch_documents


@app.command()
def index(
    index_name: str = typer.Option("tacutopapers", "--index-name", "-n"),
    df_name_or_path: str = typer.Option("hf://datasets/longevity-genie/tacutu_papers/tacutu_pubmed.parquet", "--df-name-or-path", "-d"),
    model: EmbeddingModel = typer.Option(EmbeddingModel.JINA_EMBEDDINGS_V3.value, "--model", "-m", help="Embedding model to use"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7700, "--port", "-p"),
    api_key: str = typer.Option(None, "--api-key", "-k"),
    test: bool = typer.Option(True, "--test", "-t", help="Test the index"),
    ensure_server: bool = typer.Option(True, "--ensure-server", "-e", help="Ensure Meilisearch server is running"),
    recreate_index: bool = typer.Option(False, "--recreate-index", "-r", help="Recreate the index if it already exists"),
    offset: int = typer.Option(0, "--offset", "-o", help="Offset for the index"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit for the index"),
    similarity_threshold: float = typer.Option(None, "--similarity-threshold", "-s", help="Semantic similarity threshold for the index"),
    batch_size: int = typer.Option(100, "--batch-size", "-b", help="Batch size for the index"),
) -> None:
    """Create and configure a MeiliRAG index."""
    start_index_time = time.time()  # Add this line here
    total_batches = 0  # Keep track of total batches

    if api_key is None:
        api_key = os.getenv("MEILI_MASTER_KEY", "fancy_master_key")

    
    with start_task(action_type="index_paperset", 
                    index_name=index_name, model=model, host=host, port=port, api_key=api_key, recreate_index=recreate_index, test=test, ensure_server=ensure_server) as action:
        if ensure_server:
            action.log(message_type="ensuring_server", host=host, port=port)
            ensure_meili_is_running(project_dir, host, port)
        sentence_transformer_model = load_sentence_transformer_from_enum(model)
        params = load_sentence_transformer_params_from_enum(model)
        if similarity_threshold is None:
            splitter = ArticleParagraphSplitter(model=sentence_transformer_model, batch_size=64, normalize_embeddings=False, model_params=params) 
        else:   
            splitter = ArticleSemanticParagraphSplitter(model=sentence_transformer_model, batch_size=64, normalize_embeddings=False, similarity_threshold=similarity_threshold, model_params=params) 
        config = MeiliConfig(host=host, port=port, api_key=api_key)
        rag = MeiliRAG(index_name, model, config, 
                    create_index_if_not_exists=True, 
                    recreate_index=recreate_index)
        cols = SCHOLAR_MAIN_COLUMNS
        frame = pl.read_parquet(df_name_or_path, columns=cols).slice(offset=offset, length=limit)
        df = pt.DataFrame[Paper](frame).set_model(Paper)
       
        
        # Process papers in batches using streaming
        papers_processed = 0
        current_batch = []
        
        for paper in df.iter_models():
            current_batch.append(paper)
            if len(current_batch) >= batch_size:
                process_batch(current_batch, splitter, rag)
                papers_processed += len(current_batch)
                total_batches += 1  # Increment batch counter
                current_batch = []
        
        # Process any remaining papers
        if current_batch:
            process_batch(current_batch, splitter, rag)
            papers_processed += len(current_batch)
            total_batches += 1  # Increment batch counter
            
        total_time = time.time() - start_index_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        action.add_success_fields(
            message_type="indexing_complete",
            papers_processed=papers_processed,
            total_batches=total_batches,
            start_offset=offset,
            total_time_hours=hours,
            total_time_minutes=minutes,
            total_time_seconds=seconds,
            total_time_str=f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        )



if __name__ == "__main__":
    app()

"""
Total time: 12 seconds (from 23:14:08 to 23:14:20)
Number of papers: 10
Average time per paper: 1.2 seconds
Maximum time: 3 seconds (fifth paper)
"""