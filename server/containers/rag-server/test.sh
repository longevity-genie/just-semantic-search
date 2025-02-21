#!/bin/bash

# Get current user ID and group ID
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Create necessary directories in the container
#podman
docker run -it \
    -v ./chat_agent_profiles.yaml:/app/agent_profiles.yaml \
    --network=host \
    ghcr.io/longevity-genie/just-semantic-search/rag-server:local \
    python -m just_semantic_search.server.run_rag_server run_rag_server_command