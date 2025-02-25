# syntax=docker/dockerfile:1

ARG BASE_IMAGE=ghcr.io/longevity-genie/just-agents/chat-ui-agents:main
FROM ${BASE_IMAGE}

# Create directory and add requirements

ADD server/containers/rag-server/init.py /app/init.py
ADD server/containers/rag-server/requirements.txt /app/agent_tools/requirements.txt

RUN poetry config installer.re-resolve false

# Add poetry remove command
#RUN poetry remove just-agents-web just-agents-core #we remove local version of just-agents-web because it will conflictwith just-semantic-search-server
#RUN pip freeze  2>&1
#RUN bash /usr/local/bin/entrypoint.sh echo "Installing dependencies with the entrypoint!" 2>&1
ARG JUST_SEMANTIC_SEARCH_VERSION=0.2.1
RUN poetry show && poetry add \
    just-semantic-search-meili>=$JUST_SEMANTIC_SEARCH_VERSION \
    just-semantic-search-scholar>=$JUST_SEMANTIC_SEARCH_VERSION \
    just-semantic-search-server>=$JUST_SEMANTIC_SEARCH_VERSION
# Add verification step
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || echo "PyTorch not found" 2>&1
CMD ["python", "-m", "just_semantic_search.server.rag_server", "run_rag_server_command"]
