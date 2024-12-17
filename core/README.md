# just-semantic-search

[![PyPI version](https://badge.fury.io/py/just-semantic-search.svg)](https://badge.fury.io/py/just-semantic-search)
[![Python Version](https://img.shields.io/pypi/pyversions/just-semantic-search.svg)](https://pypi.org/project/just-semantic-search/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://static.pepy.tech/badge/just-semantic-search)](https://pepy.tech/project/just-semantic-search)

LLM-agnostic semantic-search library with hybrid search support and multiple backends.

## Features

- 🔍 Hybrid search combining semantic and keyword search
- 🚀 Multiple backend support (Meilisearch, more coming soon)
- 📄 Smart document splitting with semantic awareness
- 🔌 LLM-agnostic - works with any embedding model
- 🎯 Optimized for scientific and technical content
- 🛠 Easy to use API and CLI tools

## Installation

Make sure you have at least Python 3.11 installed.

### Using pip

```bash
pip install just-semantic-search        # Core package
pip install just-semantic-search-meili  # Meilisearch backend
```

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
poetry install

# Activate the virtual environment
poetry shell
```

### Docker Setup for Meilisearch

The project includes a Docker Compose configuration for running Meilisearch. Simply run:

```bash
./bin/meili.sh
```

This will start a Meilisearch instance with vector search enabled and persistent data storage.

## Quick Start

### Document Splitting

```python
from just_semantic_search.article_semantic_splitter import ArticleSemanticSplitter
from sentence_transformers import SentenceTransformer

# Initialize model and splitter
model = SentenceTransformer('thenlper/gte-base')
splitter = ArticleSemanticSplitter(model)

# Split document with metadata
documents = splitter.split_file(
    "path/to/document.txt",
    embed=True,
    title="Document Title",
    source="https://source.url"
)
```

### Hybrid Search with Meilisearch

```python
from just_semantic_search.meili.rag import MeiliConfig, MeiliRAG

# Configure Meilisearch
config = MeiliConfig(
    host="127.0.0.1",
    port=7700,
    api_key="your_api_key"
)

# Initialize RAG
rag = MeiliRAG(
    "test_index",
    "thenlper/gte-base",
    config,
    create_index_if_not_exists=True
)

# Add documents and search
rag.add_documents_sync(documents)
results = rag.search(
    text_query="What are CAD-genes?",
    vector=model.encode("What are CAD-genes?")
)
```

## Project Structure

The project consists of multiple components:

- `core`: Core interfaces for hybrid search implementations
- `meili`: Meilisearch backend implementation


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{just_semantic_search,
  title = {just-semantic-search: LLM-agnostic semantic search library},
  author = {Karmazin, Alex and Kulaga, Anton},
  year = {2024},
  url = {https://github.com/your-username/just-semantic-search}
}
```
