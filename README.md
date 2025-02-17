# just-semantic-search

[![PyPI version](https://badge.fury.io/py/just-semantic-search.svg)](https://badge.fury.io/py/just-semantic-search)
[![Python Version](https://img.shields.io/pypi/pyversions/just-semantic-search.svg)](https://pypi.org/project/just-semantic-search/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/longevity-genie/just-semantic-search/actions/workflows/test.yml/badge.svg)](https://github.com/longevity-genie/just-semantic-search/actions/workflows/test.yml)

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

# Install Poetry Shell plugin
poetry self add poetry-plugin-shell

# Activate the virtual environment
poetry shell
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
  url = {https://github.com/your-username/just-semantic-search}
}
```