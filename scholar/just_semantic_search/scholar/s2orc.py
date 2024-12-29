import json
from typing import Optional
import typer
import requests
from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env file before initializing API_KEY
load_dotenv()

# Initialize API_KEY from environment variable
API_KEY = os.getenv('SCHOLAR_API_KEY')

# Replace Context class with simple state management
app = typer.Typer()

def get_api_key() -> str:
    if not API_KEY:
        raise typer.BadParameter("API key is required")
    return API_KEY

def make_request(url: str) -> dict:
    api_key = get_api_key()
    headers = {
        'x-api-key': api_key,
        'Accept': 'application/json'
    }
    
    # Add debug output
    typer.echo(f"Making request to: {url}")
    typer.echo(f"Using API key: {api_key[:4]}...{api_key[-4:] if api_key else ''}")
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 403:
        raise typer.BadParameter(
            "API request forbidden. Please check if your API key is valid and has the necessary permissions."
        )
    
    response.raise_for_status()
    return response.json()

def write_output(data: dict, output_path: Optional[Path]):
    if output_path:
        output_path.write_text(json.dumps(data, indent=2))
        typer.echo(f"Output written to {output_path}")
    else:
        typer.echo(data)

# Main callback to set API key
@app.callback(help="CLI tool to interact with Semantic Scholar's dataset API.", no_args_is_help=True)
def set_api_key(
    key: str = typer.Option(..., help="Semantic Scholar API key", envvar="SCHOLAR_API_KEY")
):
    """CLI tool to interact with Semantic Scholar's dataset API."""
    global API_KEY
    API_KEY = key

@app.command()
def last_three_releases(
    output: Optional[Path] = typer.Option(None, help="File path to write the output JSON.")
):
    """Fetch and print the last three release dates."""
    url = 'https://api.semanticscholar.org/datasets/v1/release'
    releases = make_request(url)
    write_output(releases[-3:], output)

@app.command("latest_release")
def latest_release(
    output: Optional[Path] = typer.Option(None, help="File path to write the output JSON.")
):
    """Fetch and print the latest release ID and details of the first dataset."""
    url = 'https://api.semanticscholar.org/datasets/v1/release/latest'
    latest = make_request(url)
    write_output(latest, output)

@app.command("dataset_info")
def dataset_info(
    dataset_name: str = typer.Argument(..., help="Name of the dataset"),
    output: Optional[Path] = typer.Option(None, help="File path to write the output JSON.")
):
    """Fetch and print info for a specific dataset in the latest release."""
    url = f'https://api.semanticscholar.org/datasets/v1/release/latest/dataset/{dataset_name}'
    dataset_info = make_request(url)
    write_output(dataset_info, output)

@app.command()
def s2orc(
    output: Optional[Path] = typer.Option(None, help="File path to write the output JSON."),
    field: Optional[str] = typer.Option(None, help="Specific field to extract from the response")
):
    """Fetch and print info for S2ORC dataset in the latest release."""
    url = 'https://api.semanticscholar.org/datasets/v1/release/latest/dataset/s2orc'
    dataset_info = make_request(url)
    result = dataset_info if field is None else dataset_info[field]
    write_output(result, output)

@app.command("s2orc_files")
def s2orc_files(
    output: Path = typer.Option("s2orc_files.txt", help="File path to write the output")
):
    """Fetch and print S2ORC filenames."""
    url = 'https://api.semanticscholar.org/datasets/v1/release/latest/dataset/s2orc'
    dataset_info = make_request(url)
    filenames = dataset_info["files"]
    with output.open('w') as file:
        for filename in filenames:
            file.write(filename + '\n')

@app.command("incremental_updates")
def incremental_updates(
    dataset_name: str = typer.Argument(..., help="Name of the dataset"),
    output: Optional[Path] = typer.Option(None, help="File path to write the output JSON.")
):
    """Fetch and print incremental updates for a specific dataset."""
    url = f'https://api.semanticscholar.org/datasets/v1/dataset/{dataset_name}/updates'
    updates = make_request(url)
    write_output(updates, output)

@app.command("release_data")
def release_data(
    release_id: str = typer.Argument(..., help="Release ID"),
    dataset_name: str = typer.Argument(..., help="Dataset name"),
    output: Optional[Path] = typer.Option(None, help="File path to write the output JSON.")
):
    """Fetch and print detailed release data for a specific dataset within a specified release."""
    url = f'https://api.semanticscholar.org/datasets/v1/release/{release_id}/dataset/{dataset_name}'
    release_data = make_request(url)
    write_output(release_data, output)

if __name__ == "__main__":
    app()