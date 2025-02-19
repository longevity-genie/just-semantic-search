#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to the MeiliSearch service directory relative to script location
cd "$SCRIPT_DIR/../services/"

# Stop and remove existing container if it exists
podman compose down
podman rm -f meilisearch 2>/dev/null || true

# Run podman compose (removing -d flag to run in foreground)
podman compose up
