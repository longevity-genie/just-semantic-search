# syntax=docker/dockerfile:1

ARG BASE_IMAGE=ghcr.io/longevity-genie/just-agents/chat-ui-agents:main
FROM ${BASE_IMAGE}

# Create directory and add requirements

ADD server/containers/rag-server/init.py /app/init.py
ADD server/containers/rag-server/requirements.txt /app/agent_tools/requirements.txt
ADD server/containers/rag-server/chat_agent_profiles.yaml /app/chat_agent_profiles.yaml

ARG JUST_SEMANTIC_SEARCH_VERSION=0.3.1
ARG JUST_AGENTS_WEB_VERSION=0.6.3

RUN poetry source add --priority explicit pytorch_cpu https://download.pytorch.org/whl/cpu \
    && poetry add --no-cache --verbose --source pytorch_cpu torch>=2.6
RUN poetry remove just_agents-web # to avoid version conflict with local version
RUN poetry add --no-cache --verbose just-agents-core==$JUST_AGENTS_WEB_VERSION just-agents-web==$JUST_AGENTS_WEB_VERSION # ugly fix to make build work
RUN poetry add --no-cache --verbose just-semantic-search-server==$JUST_SEMANTIC_SEARCH_VERSION

RUN mkdir /app/meili && chmod 777 /app/init.py /app/poetry.lock /app/meili

# Add verification step
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || echo "PyTorch not found" 2>&1
RUN python -m  just_semantic_search.embeddings 2>&1 #to preload the embeddings
CMD ["python", "-m", "just_semantic_search.server.rag_server"]
