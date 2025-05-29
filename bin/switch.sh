#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to the project root directory
cd "$SCRIPT_DIR/.."

# Check if a mode is specified
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [cpu|gpu]"
    echo "  cpu: Switch to CPU mode"
    echo "  gpu: Switch to GPU/CUDA mode"
    exit 1
fi

MODE="$1"
if [ "$MODE" != "cpu" ] && [ "$MODE" != "gpu" ]; then
    echo "Error: Mode must be either 'cpu' or 'gpu'"
    exit 1
fi

# Initialize error flag
HAS_ERRORS=0

# Function to clean up backup files
cleanup_backups() {
    local dir=$1
    cd "$SCRIPT_DIR/../$dir"
    rm -f *.bak
}

# Function to check if already in the requested mode
check_current_mode() {
    local dir=$1
    local requested_mode=$2
    local is_cuda=false
    local has_inconsistencies=false
    
    cd "$SCRIPT_DIR/../$dir"
    
    # Check package name suffix
    local name_has_cuda=false
    if grep -q 'name = ".*-cuda"' pyproject.toml; then
        name_has_cuda=true
        is_cuda=true
    fi
    
    # For core package, also check torch source and triton
    if [ "$dir" = "core" ]; then
        local torch_is_cuda=false
        local triton_is_cuda=false

        # Check torch source
        if grep -q 'torch = { version = ".*", source = "torch-gpu" }' pyproject.toml || grep -q 'torch = { version = ".*+cu[0-9]*' pyproject.toml; then
            torch_is_cuda=true
            is_cuda=true
        fi
        
        # Check triton dependency
        if ! grep -q 'triton = { version = ".*", optional = true' pyproject.toml && grep -q 'triton = { version = ".*" }' pyproject.toml; then
            triton_is_cuda=true
            is_cuda=true
        fi

        # Check for inconsistencies
        if [ "$name_has_cuda" != "$torch_is_cuda" ] || [ "$name_has_cuda" != "$triton_is_cuda" ] || [ "$torch_is_cuda" != "$triton_is_cuda" ]; then
            echo "Warning: Inconsistent GPU/CPU indicators in $dir package."
            has_inconsistencies=true
        fi
    fi
    
    # For non-core packages, check dependencies
    if [ "$dir" != "core" ]; then
        local deps_has_cuda=false
        if grep -q 'just-semantic-search-cuda' pyproject.toml; then
            deps_has_cuda=true
            is_cuda=true
        fi

        # Check for inconsistencies
        if [ "$name_has_cuda" != "$deps_has_cuda" ]; then
            echo "Warning: Inconsistent GPU/CPU indicators in $dir package."
            has_inconsistencies=true
        fi
    fi
    
    # If there are inconsistencies, force the requested mode
    if [ "$has_inconsistencies" = true ]; then
        echo "Forcing consistency for $dir to $requested_mode mode due to inconsistencies..."
        return 0
    fi
    
    # Otherwise check if mode already matches requested mode
    if [ "$requested_mode" = "gpu" ] && [ "$is_cuda" = true ]; then
        echo "Package in $dir is already in GPU mode."
        return 1
    elif [ "$requested_mode" = "cpu" ] && [ "$is_cuda" = false ]; then
        echo "Package in $dir is already in CPU mode."
        return 1
    fi
    
    # If we get here, mode doesn't match requested mode and there are no inconsistencies
    echo "Switching $dir to $requested_mode mode..."
    return 0
}

# Function to update root project dependencies
update_root_dependencies() {
    local to_cuda=$1
    cd "$SCRIPT_DIR/.."
    
    # Backup the original file
    cp pyproject.toml pyproject.toml.bak
    
    if [ "$to_cuda" = true ]; then
        # Switch to CUDA mode
        echo "Updating root project dependencies to GPU mode..."
        
        # Replace the dependencies in the main pyproject.toml
        sed -i.bak 's/just-semantic-search = { path/just-semantic-search-cuda = { path/g' pyproject.toml
        sed -i.bak 's/just-semantic-search-meili = { path/just-semantic-search-meili-cuda = { path/g' pyproject.toml
        sed -i.bak 's/just-semantic-search-scholar = { path/just-semantic-search-scholar-cuda = { path/g' pyproject.toml
        sed -i.bak 's/just-semantic-search-server = { path/just-semantic-search-server-cuda = { path/g' pyproject.toml
    else
        # Switch to CPU mode
        echo "Updating root project dependencies to CPU mode..."
        
        # Replace the dependencies in the main pyproject.toml
        sed -i.bak 's/just-semantic-search-cuda = { path/just-semantic-search = { path/g' pyproject.toml
        sed -i.bak 's/just-semantic-search-meili-cuda = { path/just-semantic-search-meili = { path/g' pyproject.toml
        sed -i.bak 's/just-semantic-search-scholar-cuda = { path/just-semantic-search-scholar = { path/g' pyproject.toml
        sed -i.bak 's/just-semantic-search-server-cuda = { path/just-semantic-search-server = { path/g' pyproject.toml
    fi
    
    # Clean up backup files
    rm -f *.bak
}

# Function to switch package mode
switch_package_mode() {
    local dir=$1
    local to_cuda=$2
    local original_name=""
    local cuda_suffix="-cuda"
    
    cd "$SCRIPT_DIR/../$dir"
    
    # Check if we need to switch
    check_current_mode "$dir" $([ "$to_cuda" = true ] && echo "gpu" || echo "cpu")
    if [ $? -ne 0 ]; then
        return 0
    fi
    
    # Get the original package name
    original_name=$(grep 'name = ' pyproject.toml | head -1 | sed -E 's/name = "([^"]*(-cuda)?)"/\1/' | sed 's/-cuda$//')
    
    if [ "$to_cuda" = true ]; then
        # Switch to CUDA mode
        cuda_name="$original_name$cuda_suffix"
        
        echo "Switching $dir to GPU mode..."
        
        # Replace the package name if it doesn't already have the CUDA suffix
        if ! grep -q "name = \"$cuda_name\"" pyproject.toml; then
            sed -i.bak "s/name = \"$original_name\"/name = \"$cuda_name\"/" pyproject.toml
        fi
        
        # Update the description to indicate CUDA version
        sed -i.bak 's/description = ".*"/description = "Core interfaces for hybrid search implementations (CUDA version)"/' pyproject.toml
        
        # Update keywords to include CUDA support
        if ! grep -q '"gpu"' pyproject.toml || ! grep -q '"cuda"' pyproject.toml; then
            # Create a proper keyword replacement that avoids duplicates
            # First get the current keywords line
            current_keywords=$(grep 'keywords = ' pyproject.toml)
            
            # Replace cpu with gpu and cuda, ensuring no duplicates
            if echo "$current_keywords" | grep -q '"cpu"'; then
                # Replace cpu with gpu and cuda
                sed -i.bak 's/"cpu"/"gpu", "cuda"/g' pyproject.toml
            else
                # Add gpu and cuda after llm if they don't exist
                sed -i.bak 's/"llm"/"llm", "gpu", "cuda"/g' pyproject.toml
            fi
            
            # Remove any duplicate consecutive keywords that might have been created
            sed -i.bak 's/, "gpu", "gpu"/, "gpu"/g; s/, "cuda", "cuda"/, "cuda"/g' pyproject.toml
        fi
        
        # Handle torch version only in core package
        if [ "$dir" = "core" ]; then
            # Extract just the version number without quotes
            TORCH_CPU_VERSION=$(grep -oP 'torch = \{ version = "\K[^"]+' pyproject.toml || echo "2.6.0")
            if [ -z "$TORCH_CPU_VERSION" ]; then
                echo "Could not detect torch version in core/pyproject.toml, defaulting to 2.6.0"
                TORCH_CPU_VERSION="2.6.0"
            fi
            
            # Remove +cu124 suffix if it already exists (to avoid adding it twice)
            CLEAN_VERSION=$(echo "$TORCH_CPU_VERSION" | sed 's/+cu124//')
            
            # Add cu124 suffix
            TORCH_CUDA_VERSION="$CLEAN_VERSION+cu124"
            
            # Perform the replacement with proper quoting
            sed -i.bak "s/torch = { version = \"[^\"]*\", source = \"torch-cpu\" }/torch = { version = \"$TORCH_CUDA_VERSION\", source = \"torch-gpu\" }/" pyproject.toml
            # If the above fails, try a more general pattern
            grep -q 'torch = { version = ".*", source = "torch-gpu" }' pyproject.toml || sed -i.bak "s/torch = { version = \"[^\"]*\".*}/torch = { version = \"$TORCH_CUDA_VERSION\", source = \"torch-gpu\" }/" pyproject.toml
            
            # Make triton a direct dependency for CUDA version
            TRITON_VERSION=$(grep -oP 'triton = \{ version = "\K[^"]+' pyproject.toml || echo ">=3.2.0")
            sed -i.bak "s/triton = { version = \"$TRITON_VERSION\", optional = true.*}/triton = { version = \"$TRITON_VERSION\" }/" pyproject.toml
            # If pattern not found, try more general replacement
            grep -q "triton = { version = \"$TRITON_VERSION\" }" pyproject.toml || sed -i.bak "s/triton = { version = \"[^\"]*\".*}/triton = { version = \"$TRITON_VERSION\" }/" pyproject.toml
        fi
        
        # Update dependencies to use CUDA versions
        if [ "$dir" != "core" ]; then
            # Update just-semantic-search dependency to use CUDA version
            sed -i.bak 's/just-semantic-search = "\*"/just-semantic-search-cuda = "\*"/' pyproject.toml
            # Also update dev dependencies if they exist
            sed -i.bak 's/just-semantic-search = { path/just-semantic-search-cuda = { path/g' pyproject.toml
            
            # For scholar and server, update meili dependency to use CUDA version
            if [ "$dir" = "scholar" ] || [ "$dir" = "server" ]; then
                sed -i.bak 's/just-semantic-search-meili = "\*"/just-semantic-search-meili-cuda = "\*"/' pyproject.toml
                sed -i.bak 's/just-semantic-search-meili = { path/just-semantic-search-meili-cuda = { path/g' pyproject.toml
            fi
            
            # For server, update scholar dependency to use CUDA version
            if [ "$dir" = "server" ]; then
                sed -i.bak 's/just-semantic-search-scholar = "\*"/just-semantic-search-scholar-cuda = "\*"/' pyproject.toml
                sed -i.bak 's/just-semantic-search-scholar = { path/just-semantic-search-scholar-cuda = { path/g' pyproject.toml
            fi
        fi
    else
        # Switch to CPU mode
        echo "Switching $dir to CPU mode..."
        
        # Replace the package name (remove -cuda suffix)
        sed -i.bak "s/name = \"$original_name$cuda_suffix\"/name = \"$original_name\"/" pyproject.toml
        
        # Update the description to remove CUDA indication
        sed -i.bak 's/description = ".*"/description = "Core interfaces for hybrid search implementations (CPU version)"/' pyproject.toml
        
        # Update keywords to remove CUDA support
        # Replace gpu and cuda with cpu, ensuring no duplicates
        if grep -q '"gpu"' pyproject.toml || grep -q '"cuda"' pyproject.toml; then
            # First replace gpu and cuda with cpu
            sed -i.bak 's/"gpu", "cuda"/"cpu"/g' pyproject.toml
            # Then handle individual cases
            sed -i.bak 's/"gpu"/"cpu"/g' pyproject.toml
            sed -i.bak 's/"cuda"/"cpu"/g' pyproject.toml
            # Remove any duplicate cpu entries that might have been created
            sed -i.bak 's/, "cpu", "cpu"/, "cpu"/g' pyproject.toml
        fi
        
        # Handle torch version only in core package
        if [ "$dir" = "core" ]; then
            # Extract just the version number without quotes
            TORCH_CUDA_VERSION=$(grep -oP 'torch = \{ version = "\K[^"]+' pyproject.toml || echo "2.6.0+cu124")
            if [ -z "$TORCH_CUDA_VERSION" ]; then
                echo "Could not detect torch version in core/pyproject.toml, defaulting to 2.6.0"
                TORCH_CUDA_VERSION="2.6.0+cu124"
            fi
            
            # Remove +cu124 suffix
            TORCH_CPU_VERSION=$(echo "$TORCH_CUDA_VERSION" | sed 's/+cu124//')
            
            # Perform the replacement with proper quoting
            sed -i.bak "s/torch = { version = \"[^\"]*\", source = \"torch-gpu\" }/torch = { version = \"$TORCH_CPU_VERSION\", source = \"torch-cpu\" }/" pyproject.toml
            # If the above fails, try a more general pattern
            grep -q 'torch = { version = ".*", source = "torch-cpu" }' pyproject.toml || sed -i.bak "s/torch = { version = \"[^\"]*\".*}/torch = { version = \"$TORCH_CPU_VERSION\", source = \"torch-cpu\" }/" pyproject.toml
            
            # Change triton back to optional dependency
            TRITON_VERSION=$(grep -oP 'triton = \{ version = "\K[^"]+' pyproject.toml || echo ">=3.2.0")
            sed -i.bak "s/triton = { version = \"$TRITON_VERSION\" }/triton = { version = \"$TRITON_VERSION\", optional = true }/" pyproject.toml
            # If pattern not found, try more general replacement
            grep -q "triton = { version = \"$TRITON_VERSION\", optional = true" pyproject.toml || sed -i.bak "s/triton = { version = \"[^\"]*\".*}/triton = { version = \"$TRITON_VERSION\", optional = true }/" pyproject.toml
        fi
        
        # Update dependencies to use CPU versions
        if [ "$dir" != "core" ]; then
            # Update just-semantic-search dependency to use CPU version
            sed -i.bak 's/just-semantic-search-cuda = "\*"/just-semantic-search = "\*"/' pyproject.toml
            # Also update dev dependencies if they exist
            sed -i.bak 's/just-semantic-search-cuda = { path/just-semantic-search = { path/g' pyproject.toml
            
            # For scholar and server, update meili dependency to use CPU version
            if [ "$dir" = "scholar" ] || [ "$dir" = "server" ]; then
                sed -i.bak 's/just-semantic-search-meili-cuda = "\*"/just-semantic-search-meili = "\*"/' pyproject.toml
                sed -i.bak 's/just-semantic-search-meili-cuda = { path/just-semantic-search-meili = { path/g' pyproject.toml
            fi
            
            # For server, update scholar dependency to use CPU version
            if [ "$dir" = "server" ]; then
                sed -i.bak 's/just-semantic-search-scholar-cuda = "\*"/just-semantic-search-scholar = "\*"/' pyproject.toml
                sed -i.bak 's/just-semantic-search-scholar-cuda = { path/just-semantic-search-scholar = { path/g' pyproject.toml
            fi
        fi
    fi
    
    # Clean up backup files
    cleanup_backups "$dir"
}

# Function to clean up lock files
clean_lock_files() {
    echo "Cleaning poetry.lock files to ensure clean dependency resolution..."
    
    # Clean root lock file
    cd "$SCRIPT_DIR/.."
    if [ -f "poetry.lock" ]; then
        rm -f poetry.lock
    fi
    
    # Clean lock files in submodules
    for pkg in "core" "meili" "scholar" "server"; do
        cd "$SCRIPT_DIR/../$pkg"
        if [ -f "poetry.lock" ]; then
            rm -f "$SCRIPT_DIR/../$pkg/poetry.lock"
        fi
    done
}

# Function to fix inconsistencies by forcing a specific mode
force_consistency() {
    local to_cuda=$1
    echo "Scanning for inconsistencies and ensuring consistent state..."
    
    # Always run the switch for all packages to ensure consistency
    for pkg in "core" "meili" "scholar" "server"; do
        check_current_mode "$pkg" $([ "$to_cuda" = true ] && echo "gpu" || echo "cpu")
        switch_package_mode "$pkg" "$to_cuda"
    done
    
    # Update root project dependencies for consistency
    update_root_dependencies "$to_cuda"
}

# List of packages to switch
packages=("core" "meili" "scholar" "server")

# Determine mode to switch to
if [ "$MODE" = "gpu" ]; then
    echo "Switching to GPU/CUDA mode..."
    # First scan for and fix any inconsistencies
    force_consistency true
else
    echo "Switching to CPU mode..."
    # First scan for and fix any inconsistencies
    force_consistency false
fi

# Clean lock files
clean_lock_files

# Final status
if [ $HAS_ERRORS -ne 0 ]; then
    echo "Completed with errors!"
    exit 1
else
    echo "All packages successfully switched to $MODE mode."
    echo "Run 'poetry install' to update your local environment."
    exit 0
fi 