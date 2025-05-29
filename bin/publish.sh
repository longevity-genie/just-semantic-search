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

# Check if switch.sh exists
if [ ! -f "$SCRIPT_DIR/switch.sh" ]; then
    echo "Error: $SCRIPT_DIR/switch.sh not found!"
    echo "Make sure the switch.sh script is available in the bin directory."
    exit 1
fi

# Initialize error flag
HAS_ERRORS=0

# Function to validate poetry setup
validate_poetry_setup() {
    echo "Validating poetry setup for dynamic versioning..."
    
    # Check if poetry-dynamic-versioning plugin is installed
    if ! poetry self show plugins | grep -q poetry-dynamic-versioning; then
        echo "Installing poetry-dynamic-versioning plugin..."
        poetry self add poetry-dynamic-versioning
        if [ $? -ne 0 ]; then
            echo "Failed to install poetry-dynamic-versioning plugin"
            return 1
        fi
    fi
    
    # Verify git is available and we're in a git repository
    if ! command -v git &> /dev/null; then
        echo "Error: git is not available"
        return 1
    fi
    
    if ! git rev-parse --git-dir &> /dev/null; then
        echo "Error: not in a git repository"
        return 1
    fi
    
    # Check if there are any git tags
    if ! git tag | grep -q .; then
        echo "Warning: No git tags found. Dynamic versioning may not work properly."
        echo "Consider creating a tag like 'git tag v0.1.0'"
    fi
    
    echo "Poetry setup validation completed successfully."
    return 0
}

# Function to build and publish a package
publish_package() {
    local dir=$1
    
    cd "$SCRIPT_DIR/../$dir"
    
    # Validate that dynamic versioning is enabled and configured
    if ! grep -q "\[tool.poetry-dynamic-versioning\]" pyproject.toml; then
        echo "Error: poetry-dynamic-versioning not configured in $dir/pyproject.toml"
        HAS_ERRORS=1
        return 1
    fi
    
    if ! grep -q "enable = true" pyproject.toml; then
        echo "Error: poetry-dynamic-versioning not enabled in $dir/pyproject.toml"
        HAS_ERRORS=1
        return 1
    fi
    
    # Debug: Show dynamic versioning configuration from pyproject.toml
    echo "=== Dynamic versioning section before build ==="
    grep -A 5 "\[tool.poetry-dynamic-versioning\]" pyproject.toml

    # Show current git tag description for debugging
    echo "Git describe:"
    git describe --tags --always
    
    # Verify that we can get a proper version
    echo "Testing dynamic version resolution..."
    poetry version
    if [ $? -ne 0 ]; then
        echo "Error: Could not resolve dynamic version for $dir"
        HAS_ERRORS=1
        return 1
    fi
    
    # Clean any existing dist directory to ensure fresh build
    if [ -d "dist" ]; then
        rm -rf dist
    fi
    
    # Build and publish
    echo "Building package..."
    poetry build
    if [ $? -ne 0 ]; then
        echo "Package build failed!"
        HAS_ERRORS=1
        return 1
    else
        echo "Publishing package..."
        poetry publish
        if [ $? -ne 0 ]; then
            echo "Package publish failed!"
            HAS_ERRORS=1
            return 1
        else
            echo "Package built and published successfully!"
            return 0
        fi
    fi
}

# List of packages to build and publish
packages=("core" "meili" "scholar" "server")

# Validate poetry setup first
echo "Starting publish process..."
validate_poetry_setup
if [ $? -ne 0 ]; then
    echo "Poetry setup validation failed!"
    exit 1
fi

# Process each package - first CPU version, then CUDA version
for pkg in "${packages[@]}"; do
    echo "Processing $pkg package..."
    
    # Switch to CPU mode
    echo "Switching to CPU mode for $pkg..."
    "$SCRIPT_DIR/switch.sh" cpu
    if [ $? -ne 0 ]; then
        echo "Failed to switch to CPU mode for $pkg!"
        HAS_ERRORS=1
        continue
    fi
    
    # Publish CPU version
    echo "Publishing CPU version of $pkg..."
    publish_package "$pkg"
    
    # Switch to GPU mode
    echo "Switching to GPU mode for $pkg..."
    "$SCRIPT_DIR/switch.sh" gpu
    if [ $? -ne 0 ]; then
        echo "Failed to switch to GPU mode for $pkg!"
        HAS_ERRORS=1
        continue
    fi
    
    # Publish GPU version
    echo "Publishing GPU version of $pkg..."
    publish_package "$pkg"
done

# Switch back to CPU mode at the end
echo "Switching back to CPU mode..."
"$SCRIPT_DIR/switch.sh" cpu

# Final error status
if [ $HAS_ERRORS -ne 0 ]; then
    echo "Completed with errors!"
    exit 1
else
    echo "All packages built and published successfully!"
    exit 0
fi
