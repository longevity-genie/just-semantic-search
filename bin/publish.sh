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

# Function to clean up backup files
cleanup_backups() {
    local dir=$1
    cd "$SCRIPT_DIR/../$dir"
    rm -f *.bak pyproject.toml.orig
}

# Function to build and publish a package
publish_package() {
    local dir=$1
    local is_cuda=$2
    local original_name=""
    local cuda_suffix=""
    
    cd "$SCRIPT_DIR/../$dir"
    
    # If CUDA version, modify pyproject.toml
    if [ "$is_cuda" = true ]; then
        cuda_suffix="-cuda"
        
        # Get the original package name and modify it for CUDA
        original_name=$(grep 'name = ' pyproject.toml | head -1 | sed 's/name = //; s/"//g; s/^[[:space:]]*//; s/[[:space:]]*$//')
        cuda_name="$original_name$cuda_suffix"
        
        # Backup the original file
        cp pyproject.toml pyproject.toml.orig
        
        # Replace the package name in pyproject.toml
        sed -i.bak "s/name = \"$original_name\"/name = \"$cuda_name\"/" pyproject.toml
        
        # Update the description to indicate CUDA version
        sed -i.bak 's/description = ".*"/description = "Core interfaces for hybrid search implementations (CUDA version)"/' pyproject.toml
        
        # Update keywords to indicate CUDA support
        sed -i.bak 's/"python", "llm"/"python", "llm", "gpu", "cuda"/' pyproject.toml
        
        # Replace the torch dependency with the CUDA version if it exists
        sed -i.bak 's/torch = { version = "2.6.0", source = "torch-cpu" }/torch = { version = "2.6.0+cu124", source = "torch-gpu" }/' pyproject.toml
        
        # Make triton a direct dependency for CUDA version
        sed -i.bak 's/triton = { version = ">=2.3.0", optional = true, markers = "extra == '\''cuda'\''" }/triton = { version = ">=2.3.0" }/' pyproject.toml
        
        # Update dependencies to use CUDA versions
        if [ "$dir" != "core" ]; then
            # Update just-semantic-search dependency to use CUDA version
            sed -i.bak 's/just-semantic-search = "\*"/just-semantic-search-cuda = "\*"/' pyproject.toml
            
            # For scholar and server, update meili dependency to use CUDA version
            if [ "$dir" = "scholar" ] || [ "$dir" = "server" ]; then
                sed -i.bak 's/just-semantic-search-meili = "\*"/just-semantic-search-meili-cuda = "\*"/' pyproject.toml
            fi
            
            # For server, update scholar dependency to use CUDA version
            if [ "$dir" = "server" ]; then
                sed -i.bak 's/just-semantic-search-scholar = "\*"/just-semantic-search-scholar-cuda = "\*"/' pyproject.toml
            fi
        fi
        
        echo "Publishing CUDA version as $cuda_name..."
    else
        echo "Publishing CPU version..."
    fi
    
    # Build and publish
    poetry build
    if [ $? -ne 0 ]; then
        echo "Package build failed!"
        HAS_ERRORS=1
    else
        poetry publish
        if [ $? -ne 0 ]; then
            echo "Package publish failed!"
            HAS_ERRORS=1
        else
            echo "Package built and published successfully!"
        fi
    fi
    
    # Restore original pyproject.toml if it exists and clean up backup files
    if [ -f "pyproject.toml.orig" ]; then
        mv pyproject.toml.orig pyproject.toml
        echo "Restored original pyproject.toml for $dir"
    fi
    cleanup_backups "$dir"
}

# List of packages to build and publish
packages=("core" "meili" "scholar" "server")

# Process each package - first CPU version, then CUDA version
for pkg in "${packages[@]}"; do
    echo "Processing $pkg package..."
    
    # Publish CPU version (default)
    publish_package "$pkg" false
    
    # Publish CUDA version
    publish_package "$pkg" true
done

# Final error status
if [ $HAS_ERRORS -ne 0 ]; then
    echo "Completed with errors!"
    exit 1
else
    echo "All packages built and published successfully!"
    exit 0
fi