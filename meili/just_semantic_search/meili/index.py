from just_semantic_search.core.embeddings import load_gte_mlm_en
import typer
import meilisearch
import os
from dotenv import load_dotenv
from just_semantic_search.meili.rag import *
import requests
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from just_semantic_search.meili.rag import *
import time
from just_semantic_search.core.text_splitter import split_text_file_semantically_annotated
from pathlib import Path
load_dotenv(override=True)
key = os.getenv("MEILI_MASTER_KEY", "fancy_master_key")

# Default embedding model constant
DEFAULT_EMBEDDING_MODEL = "allenai/specter2_aug2023refresh" #"allenai/specter2"

app = typer.Typer()



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

@app.command()
def test_query(
    query: str = typer.Argument("test", help="Search query"),
    index_name: str = typer.Option("test", help="Name of the index to search"),
    host: str = typer.Option("127.0.0.1", help="Meilisearch host"),
    port: int = typer.Option(7700, help="Meilisearch port")
):
    config = MeiliConfig(host=host, port=port, api_key=key)
    client = MeiliRAG(config)
    results = client.search(index_name, query)
    
    typer.echo(f"Search results for '{query}' in index '{index_name}':")
    for hit in results["hits"]:
        typer.echo(f"ID: {hit['id']}, Name: {hit['name']}, Description: {hit['description']}")

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
    model_name: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="Model name"),
):
    config = MeiliConfig(host=host, port=port, api_key=key)
    client = MeiliRAG(config)
    
    try:
        index = client.create_index(index_name, primary_key, model_name)
        typer.echo(f"Successfully created index '{index_name}' with primary key '{primary_key}'")
    except Exception as e:
        typer.echo(f"An error occurred while creating the index: {e}")

@app.command()
def test_together(
    index_name: str = typer.Option("test", help="Name of the index to test"),
    primary_key: str = typer.Option("id", help="Primary key field name"),
    host: str = typer.Option("127.0.0.1", help="Meilisearch host"),
    port: int = typer.Option(7700, help="Meilisearch port"),
    model_name: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="Model name"),
    query: str = typer.Option("test", help="Search query to test")
):
    """Run a complete test cycle: delete index, create index, add documents, and search."""
    config = MeiliConfig(host=host, port=port, api_key=key)
    client = MeiliRAG(config)
    
    # Delete existing index if it exists
    try:
        client.delete_index(index_name)
        typer.echo(f"Deleted existing index '{index_name}'")
    except Exception as e:
        typer.echo(f"No existing index to delete or error occurred: {e}")
    
    # Create new index
    try:
        index = client.create_index(index_name, primary_key, model_name)
        typer.echo(f"Created new index '{index_name}' with primary key '{primary_key}'")
    except Exception as e:
        typer.echo(f"Error creating index: {e}")
        return
    
    # Add test documents
    documents = [
        {"id": 1, "name": "test1", "description": "This is a test document"},
        {"id": 2, "name": "test2", "description": "This is another test document"},
    ]
    
    count = client.add_documents(index_name, documents)
    typer.echo(f"Added {count} documents to the index")
    
    # Wait a moment for indexing
    typer.echo("Waiting for documents to be indexed...")
    time.sleep(1)
    
    # Test search
    results = client.search(index_name, query)
    typer.echo(f"\nSearch results for '{query}' in index '{index_name}':")
    for hit in results["hits"]:
        typer.echo(f"ID: {hit['id']}, Name: {hit['name']}, Description: {hit['description']}")

if __name__ == "__main__":
    app()