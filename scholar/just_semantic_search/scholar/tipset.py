from datetime import datetime
from typing import TypeVar
import typer
import polars as pl
from pathlib import Path
from typing import TypeVar
from just_semantic_search.paragraph_splitters import *
import patito as pt
from pycomfort.logging import to_nice_file, to_nice_stdout
from just_semantic_search.embeddings import *
from just_semantic_search.utils.tokens import *
from pathlib import Path
from just_semantic_search.paragraph_splitters import *
from just_semantic_search.text_splitters import *


import typer
import os
from just_semantic_search.meili.rag import *
from just_semantic_search.meili.rag import *
import time
from pathlib import Path

from eliot._output import *
from eliot import start_task
from just_semantic_search.meili.utils.services import ensure_meili_is_running




project_dir = Path(__file__).parent.parent.parent.parent
print(f"project_dir: {project_dir}")
data_dir = project_dir / "data"
logs = project_dir / "logs"
meili_service_dir = project_dir / "meili"

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

data = Path(__file__).parent.parent.parent / "data"

class Tip(pt.Model):
    title: str
    paragraph: str
    page_idx: int
    
    @property
    def abstract(self) -> str:
        return self.annotations_abstract[0] if self.annotations_abstract else None
    
    

    def to_document(self) -> Document:
        # Create metadata dictionary with all available fields
        metadata = {
            'title': self.title,
            'paragraph': self.paragraph,
            'page_idx': self.page_idx
        }
        
        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}
        return Document(
            title=self.title,
            abstract=None,
            fragment_num=0,
            total_fragments=1,
            text=self.paragraph,
            metadata=metadata,
            source=f"{self.title}_PAGE_{self.page_idx}"
        )



def process_batch(df: pl.DataFrame, splitter: DocumentParagraphSplitter, rag: MeiliRAG):
            """
            Processes batch of papers to be added to the database"""
            start_time = time.time()  # Start timing
            with start_action(action_type="process batch") as action:
                paper_start = time.time()  # Time individual paper
                paragraphs = df.select(pl.col("paragraph")).to_series().to_list()
                print(paragraphs)
                source = df.select(pl.col("title")).row(0)[0]  # Get title from first row
                documents = splitter.split(paragraphs, source=source)
                token_counts = sum([d.token_count for d in documents])
                paper_time = time.time() - paper_start
                action.log(message_type="paper_processing_time", 
                            paper_id=source, 
                            time=paper_time)
                rag.add_documents(documents=documents)
                
                batch_time = time.time() - start_time
                action.add_success_fields(
                    batch_size=len(documents),
                    batch_processing_time=batch_time,
                    total_tokens=token_counts
                )
            return documents

def data_frames(parsed: Path = Path("/home/antonkulaga/sources/just-semantic-search/data/productivity/parsed")) -> list[pl.DataFrame]:
    from pycomfort.files import dirs, with_ext
    papers = dirs(parsed).filter(lambda f: (f / "auto").exists()).map(lambda f: with_ext(f / "auto", "json").filter(lambda f: "content" in f.name).first()).to_list()
    return [pl.read_json(p,infer_schema_length=1000).filter(pl.col("text") != "").select(pl.lit(p.name.replace("_content_list.json", "")).alias("title"), pl.col("text").alias("paragraph"), pl.col("page_idx").alias("page_idx")) for p in papers]



@app.command()
def index(
    index_name: str = typer.Option("productivity", "--index-name", "-n"),
    parsed: Path = typer.Option("/home/antonkulaga/sources/just-semantic-search/data/productivity/parsed", "--parsed", "-p"),
    model: EmbeddingModel = typer.Option(EmbeddingModel.JINA_EMBEDDINGS_V3.value, "--model", "-m", help="Embedding model to use"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7700, "--port", "-p"),
    api_key: str = typer.Option(None, "--api-key", "-k"),
    test: bool = typer.Option(True, "--test", "-t", help="Test the index"),
    ensure_server: bool = typer.Option(True, "--ensure-server", "-e", help="Ensure Meilisearch server is running"),
    recreate_index: bool = typer.Option(False, "--recreate-index", "-r", help="Recreate the index if it already exists"),
    similarity_threshold: float = typer.Option(0.8, "--similarity-threshold", "-s", help="Semantic similarity threshold for the index")
) -> None:
    """Create and configure a MeiliRAG index."""
    start_index_time = time.time()
    total_batches = 0

    if api_key is None:
        api_key = os.getenv("MEILI_MASTER_KEY", "fancy_master_key")

    transformer_model = load_sentence_transformer_from_enum(model)
    
    with start_task(action_type="index_paperset", 
                    index_name=index_name, model_name=model, host=host, port=port, api_key=api_key, recreate_index=recreate_index, test=test, ensure_server=ensure_server) as action:
        if ensure_server:
            action.log(message_type="ensuring_server", host=host, port=port)
            ensure_meili_is_running(project_dir, host, port)

        splitter = ParagraphSemanticDocumentSplitter(model=transformer_model, batch_size=64, normalize_embeddings=False, similarity_threshold=similarity_threshold)
        
        rag = MeiliRAG.get_instance(
            index_name=index_name,
            model=model,
            host=host,
            port=port,
            api_key=api_key,
            create_index_if_not_exists=True,
            recreate_index=recreate_index
        )
        
        dfs = data_frames(parsed)
        for df in dfs:
            process_batch(df, splitter, rag)
            total_batches += 1
            
        total_time = time.time() - start_index_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        action.add_success_fields(
            message_type="indexing_complete",
            total_batches=total_batches,
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