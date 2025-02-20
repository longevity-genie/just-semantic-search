#!/bin/bash
set -e

# Resolve the project root directory
PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))

docker run --network host --user $(id -u):$(id -g) \
  -v ${PROJECT_ROOT}/chat_agent_profiles.yaml:/app/chat_agent_profiles.yaml \
  ghcr.io/longevity-genie/just-semantic-search/rag-server:main