from just_semantic_search.embeddings import load_gte_mlm_en
import typer
import os
from dotenv import load_dotenv
from just_semantic_search.meili.rag import *
import requests
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from just_semantic_search.meili.rag import *
import time
from pathlib import Path
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from just_semantic_search.embeddings import DEFAULT_EMBEDDING_MODEL_NAME

load_dotenv(override=True)
key = os.getenv("MEILI_MASTER_KEY", "fancy_master_key")

app = typer.Typer()
"""
@app.command()
def add_documents(
    index_name: str = typer.Option("test", help="Name of the index to add documents to"),
    host: str = typer.Option("127.0.0.1", help="Meilisearch host"),
    port: int = typer.Option(7700, help="Meilisearch port")
):
    config = MeiliConfig(host=host, port=port, api_key=key)
    client = MeiliRAG(config)
    

    file = Path("/home/antonkulaga/sources/just_semantic_search/data/tacutopapers_test_rsids_10k/108.txt")
    model = load_gte_mlm_en()
    documents = split_text_file_semantically_annotated(file, model, similarity_threshold=0.92, source="/home/antonkulaga/sources/just_semantic_search/data/tacutopapers_test_rsids_10k/108.txt")
    count = client.add_documents(index_name, documents)
    typer.echo(f"Added {count} documents to the '{index_name}' index.")
"""

@app.command()
def test_query(
    query: str = typer.Argument("test", help="Search query"),
    index_name: str = typer.Option("test", help="Name of the index to search"),
    host: str = typer.Option("127.0.0.1", help="Meilisearch host"),
    port: int = typer.Option(7700, help="Meilisearch port")
):
    config = MeiliConfig(host=host, port=port, api_key=key)
    client = MeiliRAG(config)
    
    with Console.status("[bold green]Searching..."):
        results = client.search(index_name, query)
    
    table = Table(title=f"Search Results for '{query}' in '{index_name}'")
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Description", style="green")
    
    for hit in results["hits"]:
        table.add_row(
            str(hit['id']),
            hit['name'],
            hit['description']
        )
    
    Console.print(table)

@app.command()
def delete_index(
    index_name: str = typer.Option("test", help="Name of the index to delete"),
    host: str = typer.Option("127.0.0.1", help="Meilisearch host"),
    port: int = typer.Option(7700, help="Meilisearch port")
):
    config = MeiliConfig(host=host, port=port, api_key=key)
    client = MeiliRAG(config)
    
    try:
        
        client.delete_index(index_name)
        typer.echo(f"Successfully deleted the '{index_name}' index.")
    except Exception as e:
        typer.echo(f"An error occurred while deleting the index: {e}")

@app.command()
def add_index(
    index_name: str = typer.Option("test", help="Name of the index to create"),
    primary_key: str = typer.Option("id", help="Primary key field name"),
    host: str = typer.Option("127.0.0.1", help="Meilisearch host"),
    port: int = typer.Option(7700, help="Meilisearch port"),
    model_name: str = typer.Option(DEFAULT_EMBEDDING_MODEL_NAME, help="Model name"),
):
    config = MeiliConfig(host=host, port=port, api_key=key)
    client = MeiliRAG(config)
    
    try:
        index = client.create_index(index_name, primary_key, model_name)
        typer.echo(f"Successfully created index '{index_name}' with primary key '{primary_key}'")
    except Exception as e:
        typer.echo(f"An error occurred while creating the index: {e}")


if __name__ == "__main__":
    app()