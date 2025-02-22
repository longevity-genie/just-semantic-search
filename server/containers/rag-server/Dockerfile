# syntax=docker/dockerfile:1

ARG BASE_IMAGE=ghcr.io/longevity-genie/just-agents/chat-ui-agents:main
FROM ${BASE_IMAGE}

# Create directory and add requirements
RUN mkdir -p /app/agent_tools/ /app/env /app/tmp
RUN chmod -R 777 /app
ADD server/containers/rag-server/init.py /app/agent_tools/init.py
ADD server/containers/rag-server/requirements.txt /app/agent_tools/requirements.txt
RUN ls -la /app/agent_tools/

# Add poetry remove command
RUN poetry remove just-agents-web # we remove local version of just-agents-web because it will conflictwith just-semantic-search-server

#RUN pip freeze  2>&1
RUN bash /usr/local/bin/entrypoint.sh echo "Installing dependencies with the entrypoint!" 2>&1

# Add verification step
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || echo "PyTorch not found" 2>&1
CMD ["python", "-m", "just_semantic_search.rag_server", "run_rag_server_command"]