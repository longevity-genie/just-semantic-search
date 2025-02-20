#!/bin/bash

# Get current user ID and group ID
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Create necessary directories in the container
docker run -it \
    --user $USER_ID:$GROUP_ID \
    ghcr.io/longevity-genie/just-agents/chat-ui-agents:main \
    /bin/bash -c "mkdir -p /app/agent_tools/ && mkdir -p /app/env"

# Copy requirements.txt into the container
docker cp server/containers/rag-server/requirements.txt \
    $(docker ps -lq):/app/agent_tools/requirements.txt

# Run the Docker container with entrypoint overridden to bash
docker run -it \
    --user $USER_ID:$GROUP_ID \
    -v "$(pwd)/server/containers/rag-server:/app/agent_tools" \
    -v "$(pwd)/env:/app/env" \
    --entrypoint=/bin/bash \
    ghcr.io/longevity-genie/just-agents/chat-ui-agents:main
