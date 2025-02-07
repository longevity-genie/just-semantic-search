# syntax=docker/dockerfile:1
ARG base_image=ghcr.io/longevity-genie/just-agents:main-gpu
FROM ${base_image}

USER root

# Check if conda Python exists and set Python path accordingly
RUN if [ -f "/opt/conda/bin/python" ]; then \
    echo "Using conda Python" && \
    export PATH="/opt/conda/bin:${PATH}" && \
    PYTHON_CMD="/opt/conda/bin/python3"; \
    else \
    echo "Using system Python" && \
    PYTHON_CMD="python3"; \
    fi && \
    curl -sSL https://install.python-poetry.org | $PYTHON_CMD -

ARG JUST_SEMANTIC_SEARCH_AGENT_VERSION=0.1.1

# Configure poetry without virtualenvs
RUN mkdir -p /root/.cache/pypoetry && \
    /root/.local/bin/poetry config virtualenvs.create false

# Install just-semantic-search-agent using poetry without creating a virtual environment
RUN /root/.local/bin/poetry add "just-semantic-search-agent==${JUST_SEMANTIC_SEARCH_AGENT_VERSION}" --python ">=3.11,<3.14"
RUN /root/.local/bin/poetry lock
RUN /root/.local/bin/poetry install

USER appuser

# how many attempts to retry
ENV RETRY_ATTEMPTS=5

# how much to multiply the delay between attempts
ENV RETRY_MULTIPLIER=1

# minimum delay between attempts
ENV RETRY_MIN=4

# maximum delay between attempts
ENV RETRY_MAX=10

#default host for meilisearch
ENV MEILISEARCH_HOST=0.0.0.0 

#default port for meilisearch
ENV MEILISEARCH_PORT=7700

# what is the ratio of semantic search in results ranking
ENV MEILISEARCH_SEMANTIC_RATIO=0.5

# how many documents to return
ENV MEILISEARCH_LIMIT=100

# show matches position
ENV MEILISEARCH_SHOW_MATCHES_POSITION=False

# matching strategy
ENV MEILISEARCH_MATCHING_STRATEGY=last

# show ranking score
ENV MEILISEARCH_SHOW_RANKING_SCORE=True

# show ranking score details
ENV MEILISEARCH_SHOW_RANKING_SCORE_DETAILS=True

# if we create index if it does not exist
ENV MEILISEARCH_CREATE_INDEX_IF_NOT_EXISTS=True

# if we recreate index from scratch
ENV MEILISEARCH_RECREATE_INDEX=False

ENV PARSING_RECREATE_MEILI_INDEX=False

#used for markdown parsing, characters to extract abstract from
ENV MEILISEARCH_CROP_LENGTH=1000

WORKDIR /app
