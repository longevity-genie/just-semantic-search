#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to the project root directory
cd "$SCRIPT_DIR/.."

# Ensure poetry is installed and available
if ! command -v poetry &> /dev/null; then
    echo "Poetry is not installed. Please install it first."
    echo "Visit https://python-poetry.org/docs/#installation for installation instructions."
    exit 1
fi

# Build core package first
echo "Building core package..."
cd core
poetry build
if [ $? -ne 0 ]; then
    echo "Core package build failed!"
    exit 1
fi
echo "Core package built successfully!"

# Build meili package next
echo "Building meili package..."
cd ../meili
poetry build
if [ $? -ne 0 ]; then
    echo "Meili package build failed!"
    exit 1
fi
echo "Meili package built successfully!"