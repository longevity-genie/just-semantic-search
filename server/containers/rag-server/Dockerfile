# syntax=docker/dockerfile:1

ARG BASE_IMAGE=ghcr.io/longevity-genie/just-agents/chat-ui-agents:main
FROM ${BASE_IMAGE}

# Create directory and add requirements
RUN mkdir -p /app/agent_tools/ 
RUN mkdir -p /app/env
ADD server/containers/rag-server/requirements.txt /app/agent_tools/requirements.txt
RUN ls -la /app/agent_tools/
RUN /usr/local/bin/entrypoint.sh echo "Installing dependencies with the entrypoint!"

# Add verification step
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || echo "PyTorch not found"
CMD ["python", "-m", "just_semantic_search.run_rag_server", "run_rag_server_command"]