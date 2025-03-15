#!/usr/bin/env bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse command line arguments
CACHE_FLAG=""
if [ "$1" == "--no-cache" ]; then
    CACHE_FLAG="--no-cache"
fi

# Build the Docker image
docker build $CACHE_FLAG \
    -f "${PROJECT_ROOT}/server/containers/rag-server/Dockerfile" \
    -t ghcr.io/longevity-genie/just-semantic-search/rag-server:main \
    "${PROJECT_ROOT}"
