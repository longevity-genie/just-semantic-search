# just-semantic-search
LLM-agnostic semantic-search library with hybrid search support and multiple backends.

## Installation

### Using Poetry

1. Add to your project:

```poetry add just-semantic-search-project

```

2. Or install from source:

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

3. The project consists of multiple components:
- `core`: Core interfaces for hybrid search implementations
- `meili`: Meilisearch backend implementation

Each component can be installed separately if needed:

```bash
poetry add just-semantic-search-core
poetry add just-semantic-search-meili
```

### Requirements

- Python 3.11 or higher
- Poetry 1.0.0 or higher
