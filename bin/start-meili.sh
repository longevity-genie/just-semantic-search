#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to the MeiliSearch service directory relative to script location
cd "$SCRIPT_DIR/../services/meili"

# Stop and remove existing container if it exists
docker compose down
docker rm -f meilisearch 2>/dev/null || true

# Run docker-compose (removing -d flag to run in foreground)
docker compose up
