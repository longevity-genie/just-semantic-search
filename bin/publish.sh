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

# Initialize error flag
HAS_ERRORS=0

# Build and publish core package first
echo "Building and publishing core package..."
cd core
poetry build
if [ $? -ne 0 ]; then
    echo "Core package build failed!"
    HAS_ERRORS=1
else
    poetry publish
    if [ $? -ne 0 ]; then
        echo "Core package publish failed!"
        HAS_ERRORS=1
    else
        echo "Core package built and published successfully!"
    fi
fi

# Build and publish meili package next
echo "Building and publishing meili package..."
cd ../meili
poetry build
if [ $? -ne 0 ]; then
    echo "Meili package build failed!"
    HAS_ERRORS=1
else
    poetry publish
    if [ $? -ne 0 ]; then
        echo "Meili package publish failed!"
        HAS_ERRORS=1
    else
        echo "Meili package built and published successfully!"
    fi
fi

# Final error status
if [ $HAS_ERRORS -ne 0 ]; then
    echo "Completed with errors!"
    exit 1
else
    echo "All packages built and published successfully!"
    exit 0
fi