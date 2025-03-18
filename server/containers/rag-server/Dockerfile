# syntax=docker/dockerfile:1

FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    gosu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN export POETRY_VERSION="2.1.1" && \
    mkdir -p /opt/poetry && \
    curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry && \
    export PATH="/opt/poetry/bin:$PATH" && \
    echo 'export PATH="/opt/poetry/bin:$PATH"' >> /etc/profile && \
    echo 'export PATH="/opt/poetry/bin:$PATH"' >> /etc/bash.bashrc && \
    export POETRY_HOME="/opt/poetry"

# Add Poetry to PATH
ENV POETRY_HOME="/opt/poetry"
ENV PATH="/opt/poetry/bin:$PATH"

ARG JUST_SEMANTIC_SEARCH_VERSION=0.3.6
ARG JUST_AGENTS_WEB_VERSION=0.6.8

# Create necessary directories
RUN export CONTAINER_DIRS="/app /app/models.d /app/agent_tools /app/scripts /app/data /app/meili /app/logs /app/tmp /app/env" && \
    mkdir -p $CONTAINER_DIRS && \
    chmod 777 $CONTAINER_DIRS

# Set up the app directory
WORKDIR /app
COPY ./server/containers/rag-server/requirements.txt /app/agent_tools/requirements.txt
COPY ./server/containers/rag-server/chat_agent_profiles.yaml /app/chat_agent_profiles.yaml
COPY ./server/containers/rag-server/pyproject.toml.template /app/pyproject.toml
COPY ./LICENSE.md /app/LICENSE.md
COPY ./README.md /app/README.md

# Install dependencies
#RUN poetry source add --priority explicit pytorch_cpu https://download.pytorch.org/whl/cpu \
#    && poetry add --no-cache --verbose --source pytorch_cpu torch>=2.6
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-cache
RUN poetry add --no-cache --verbose just-agents>=$JUST_AGENTS_WEB_VERSION just-semantic-search-server>=$JUST_SEMANTIC_SEARCH_VERSION && \
    poetry install --no-interaction --no-cache --compile && \
    poetry show --tree

# Copy entrypoint script and make it executable
COPY ./server/containers/rag-server/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 777 /app -R && \
    echo '#!/usr/bin/env bash' >> /app/init.py && \
    echo 'print("Preparation complete!")' >> /app/init.py && \
    chmod 777 /usr/local/bin/entrypoint.sh /app/init.py /app/poetry.lock

# Expose ports and set environment variables
EXPOSE 8091

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1


## Add verification step
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || echo "PyTorch not found" 2>&1 

# Create a specific directory for model cache and set permissions
RUN mkdir -p /app/model_cache && chmod 777 /app/model_cache

# Set the Hugging Face cache directory
ENV TRANSFORMERS_CACHE="/app/model_cache"
ENV HF_HOME="/app/model_cache"

# Preload models with the correct cache location
RUN TRANSFORMERS_CACHE="/app/model_cache" HF_HOME="/app/model_cache" python -m just_semantic_search.embeddings 2>&1

# WebAgentConfig environment variables
ENV APP_HOST="0.0.0.0"
ENV APP_PORT=8091
ENV AGENT_WORKERS=4
ENV AGENT_TITLE="Semantic Search Rag Agent Server REST API endpoint"
ENV AGENT_SECTION=""
ENV AGENT_PARENT_SECTION=""
ENV AGENT_FAILFAST="true"
ENV AGENT_DEBUG="true"
ENV AGENT_REMOVE_SYSTEM_PROMPT="false"
ENV AGENT_CONFIG_PATH="agent_profiles.yaml"
ENV ENV_KEYS_PATH="env/.env.local"
ENV APP_DIR="/app"
ENV TMP_DIR="tmp"
ENV LOG_DIR="logs"
ENV DATA_DIR="data"

# ChatUIAgentConfig additional environment variables (aligned with config.py)
ENV MODELS_DIR="models.d"
ENV ENV_MODELS_PATH="env/.env.local"
ENV REMOVE_DD_CONFIGS="true"
ENV TRAP_CHAT_NAMES="true"
ENV AGENT_HOST="http://127.0.0.1"
ENV AGENT_PORT=8091
ENV JSON_FILE_PATTERN="[0123456789][0123456789]_*.json"

# Keep running as root - let entrypoint handle user switching
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
#CMD ["python", "-m", "just_agents.web.run_agent", "run-chat-ui-server-command"]
CMD python -m just_semantic_search.server.run_rag_server --workers $AGENT_WORKERS --host $APP_HOST --port $APP_PORT

