#!/bin/bash

#init script for docker container, not yet used right now
# the idea is to have it to index everything in the folder

# Check if both arguments are provided
if [ $# -ne 2 ]; then
    echo "Error: Please provide both the markdown directory path and index name"
    echo "Usage: $0 <markdown_directory_path> <index_name>"
    exit 1
fi

# Store the markdown directory path and index name from arguments
MARKDOWN_DIR="$1"
INDEX_NAME="$2"

# Check if the directory exists
if [ ! -d "$MARKDOWN_DIR" ]; then
    echo "Error: Directory '$MARKDOWN_DIR' does not exist"
    exit 1
fi

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Error: Poetry is not installed"
    exit 1
fi

# Ensure we're in the correct poetry environment and run the indexing script
if ! poetry run python -m just_semantic_search.server.index_markdown "$MARKDOWN_DIR" --index-name "$INDEX_NAME"; then
    echo "Error: Failed to run the indexing script"
    exit 1
fi
