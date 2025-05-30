# just-semantic-search

[![PyPI version](https://badge.fury.io/py/just-semantic-search.svg)](https://badge.fury.io/py/just-semantic-search)
[![Python Version](https://img.shields.io/pypi/pyversions/just-semantic-search.svg)](https://pypi.org/project/just-semantic-search/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/longevity-genie/just-semantic-search/actions/workflows/test.yml/badge.svg)](https://github.com/longevity-genie/just-semantic-search/actions/workflows/test.yml)

LLM-agnostic semantic-search library with hybrid search support and multiple backends.
It also includes a REST API for semantic search which allows both meili-based search and agentic search.

# Using Just Semantic Search Server

We provide a REST API for semantic search that supports both Meilisearch-based and agentic search capabilities. The server includes:

- Meilisearch backend for vector and hybrid search
- RAG server with sentence-transformers for embeddings (CUDA-enabled but works without GPU)
- API endpoints for document indexing and search
- Optional Gemini integration for document summarization

## Prerequisites

- Docker or Podman installed
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (optional, for GPU support)
- At least 14GB free disk space (for container images)

## Quick Start

1. Copy and configure environment variables:

```bash
cp ./.env.template ./.env  
mkdir -p env && cp .env env/.env.local
# Then edit .env with your settings
```

2. Start the services:

```bash
podman compose up   # or 'docker compose up'
```

3. Access the services:
- REST API documentation: http://localhost:8090/docs
- Meilisearch dashboard: http://localhost:7700

## Indexing Documents

The server provides several ways to index documents:

1. Using the REST API directly (see API docs)
2. Using the `index-markdown` command for markdown files with optional Gemini summarization
3. Using the PDF conversion utility (requires GPU, should be run with sudo):

```bash
sudo server/bin/parse-papers.sh -i /path/to/pdfs -o /path/to/output
```

# Using just-semantic-search as a library

## Features

- 🔍 Hybrid search combining semantic and keyword search
- 🚀 Multiple backend support (Meilisearch, more coming soon)
- 📄 Smart document splitting with semantic awareness
- 🔌 LLM-agnostic - works with any embedding model
- 🎯 Optimized for scientific and technical content
- 🛠 Easy to use API and CLI tools

## Installation

Make sure you have at least Python 3.10 installed (3.11 and higher is recommended).

### Using pip


The packages come in two flavors:
- Standard CPU version: `pip install just-semantic-search`
- CUDA-enabled version: `pip install just-semantic-search-cuda`

For meilisearch backend you need to install:

```bash
pip install just-semantic-search-meili  # Meilisearch backend
```

### GPU and CPU Versions

All subprojects within just-semantic-search are published in two flavors:
- Standard CPU version (e.g., `just-semantic-search`, `just-semantic-search-meili`)
- CUDA-enabled GPU version (e.g., `just-semantic-search-cuda`, `just-semantic-search-meili-cuda`)

If you don't have CUDA-compatible hardware or don't need GPU acceleration, use the standard CPU versions. For those with compatible GPUs who want to leverage hardware acceleration, choose the CUDA-enabled versions.

#### Switching Between Versions

When developing with the library from source, you can easily switch between CPU and GPU versions using the provided script:

```bash
# Switch to CPU mode
./bin/switch.sh cpu

# Switch to GPU/CUDA mode
./bin/switch.sh gpu

# After switching, update your environment
poetry install
```

This will update all subprojects and dependencies to use either the CPU or GPU versions, allowing you to choose the best configuration for your hardware.

### Using Poetry

```bash
poetry add just-semantic-search        # Core package
poetry add just-semantic-search-meili  # Meilisearch backend
```

### From Source

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone https://github.com/your-username/just-semantic-search.git
cd just-semantic-search

# Install dependencies and create virtual environment
poetry install #cpu by default, to install gpu version use poetry install --extras cuda

# Install Poetry Shell plugin
poetry self add poetry-plugin-shell

# Activate the virtual environment
poetry shell
```

Note: to make proper GPU (CUDA) resolution you must apply:
```bash
poetry config installer.re-resolve false
poetry install --extras cuda
```


### Docker and Podman Setup for Meilisearch

The project includes a Docker Compose configuration for running Meilisearch as a service to make it easier to run it locally. 
Simply run:

```bash
./bin/meili.sh
```

This will start a Meilisearch instance with vector search enabled and persistent data storage.

**Podman Users:**

If you prefer using Podman over Docker, you can use `podman compose` as a drop-in replacement. In many cases, the same configuration works, so you can start the service with:

```bash
podman compose up
```

Make sure that `podman compose` is installed and properly configured on your system.

WARNING: on old Ubuntu systems (like Ubuntu 22.04 and older) do not install podman from apt, as it is outdated there. You need at least podman 4+ to make it work. 

## Quick Start

### Document Splitting

```python
from just_semantic_search.article_semantic_splitter import ArticleSemanticSplitter
from just_semantic_search.embeddings import EmbeddingModel, load_sentence_transformer_from_enum

# Initialize model using the embeddings enum (using Jina v3 by default)
model = load_sentence_transformer_from_enum(EmbeddingModel.JINA_EMBEDDINGS_V3) #you can also use SentenceTransformer directly, we provided enums just for most popular ones
splitter = ArticleSemanticSplitter(model)

# Split document with metadata - embeddings will automatically use optimal parameters for passage encoding
documents = splitter.split_file(
    "path/to/document.txt",
    embed=True,
    title="Document Title",
    source="https://source.url"
)
```

### Hybrid Search with Meilisearch

```python
from just_semantic_search.meili.rag import MeiliRAG
from just_semantic_search.embeddings import EmbeddingModel

# Initialize RAG with Jina embeddings
rag = MeiliRAG(
    index_name="test_index",
    model=EmbeddingModel.JINA_EMBEDDINGS_V3,  # Uses optimal parameters for different tasks
    host="127.0.0.1",
    port=7700,
    api_key="your_api_key",
    create_index_if_not_exists=True
)

# Add documents and search
rag.add_documents(documents)

# The search will automatically use the optimal query encoding parameters
results = rag.search(
    query="What are CAD-genes?",
    semanticRatio=0.5  # Adjust the ratio between semantic and keyword search
)
```

The library uses Jina embeddings v3 by default, which automatically optimizes embedding parameters for different tasks:
- Query encoding for search queries
- Passage encoding for document content
- Text matching for similarity comparisons
- Classification for categorization tasks
- Separation for clustering and reranking

### Remote and Local Embeddings

The library supports both local and remote embeddings processing:
- **Local embeddings**: Runs on your machine with CPU or GPU acceleration (if available)
- **Remote embeddings**: Uses cloud-based API services for potentially better performance without local resource consumption
- **Reranking support**: For Jina models, enabling more accurate search result ordering

To use remote embeddings, you need to configure your API key in the environment:

```bash
# For remote embeddings
export JINAAI_API_KEY="your_api_key_here"
```

Without an API key, the library will default to local embedding processing.

Other available models include:
- `EmbeddingModel.GTE_LARGE` - General Text Embeddings (large)
- `EmbeddingModel.BIOEMBEDDINGS` - Specialized for biomedical text
- `EmbeddingModel.SPECTER` - Optimized for scientific papers
- `EmbeddingModel.MEDCPT_QUERY` - Medical domain queries

## Project Structure

The project consists of multiple components:

- `core`: Core interfaces for hybrid search implementations
- `meili`: Meilisearch backend implementation


## Testing

### Manual testing

For manual testing we provide manual.py script that allows to test different cases.

To fill in the text index you should use index-folder command.

```bash
poetry shell
python test/meili/manual.py index-folder
```

After the index is filled you can test the search cases.
The index-folder command checks if meili serve is available and if not starts it by running ./bin/meili.sh under the hood.
You can also add --test flat to run rsid and superhero search tests

#### Test search

```bash
python test/meili/manual.py test-search
```
This test requires test folder to be indexed (see index-folder command) and contains two cases:
1. RSID test
2. Comics superheroes test

##### RSID test
There are text pieces deliberately incorporated into tacutu papers data ( /data/tacutopapers_test_rsids_10k ) In particular for rs123456789 and rs123456788 as well as similar but misspelled rsids are added to the documents:

10.txt contains both two times
11.txt contains both one time
12.txt and 13 contain only one rsid
20.txt contains both wrong rsids two times
21.txt contains both wrong rsids one time
22.txt and 23 contain only one wrong rsid

##### Comics superheroes test
Also, similar test for "Comics superheroes" that will test embeddings:

Only 114 document has text about superheroes, but text did not contain words 'comics' or 'superheroes'

## Indexing example hugginfa S2ORC datasets

We uploaded many papers from S2ORC dataset as parquet files. As those papers are already split into paragraphs, we use paragraph splitters.
You can try it out with the following command:

```bash
poetry shell
python scholar/just_semantic_search/scholar/paperset.py index --index-name paperset --df-name-or-path hf://datasets/longevity-genie/tacutu_papers/tacutu_pubmed.parquet --model-name gte-large
```

By default it does not use semantic similarity threshold so it will split by maximum token length. However, you can specify similarity threshold as -s parameters to split by semantic similarity but it is very slow.
If you add --recreate-index it will clean the index before adding which is useful for testing.

You can also try to index all ageing research papers from Semantic Scholar dataset but it will take days to complete:
```bash
python scholar/just_semantic_search/scholar/paperset.py index --recreate-index --index-name paperset --df-name-or-path hf://datasets/longevity-genie/aging_papers_paragraphs/aging_specific_pubmed.parquet --model-name gte-large
```


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE.md) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{just_semantic_search,
  title = {just-semantic-search: LLM-agnostic semantic search library},
  author = {Karmazin, Alex and Kulaga, Anton},
  year = {2024},
  url = {https://github.com/longevity-genie/just-semantic-search}
}
```
