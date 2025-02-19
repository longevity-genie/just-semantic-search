#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Build Dockerfile from parent directory relative to script location with specific label
docker build -f "${SCRIPT_DIR}/../Dockerfile" "${SCRIPT_DIR}/.." -t ghcr.io/longevity-genie/just-semantic-search/server:main
