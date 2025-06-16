#!/bin/bash
set -o pipefail

# Logging functions
log_message() {
    local message="$1"
    if [[ -n "${message// }" ]]; then
        echo "[LAUNCHER_LOGGER] [MODEL_INSTALL_LLAMA] --message \"$message\""
    fi
}

log_error() {
    local message="$1"
    if [[ -n "${message// }" ]]; then
        echo "[LAUNCHER_LOGGER] [MODEL_INSTALL_LLAMA] --error \"$message\"" >&2
    fi
}

# Error handling function
handle_error() {
    local exit_code=$1
    local error_msg=$2
    log_error "$error_msg (Exit code: $exit_code)"
    
    # Clean up if needed
    if [[ -n "$VIRTUAL_ENV" ]]; then
        log_message "Deactivating virtual environment..."
        deactivate 2>/dev/null || true
    fi
    
    exit $exit_code
}

command_exists() {
    command -v "$1" &> /dev/null
}

# Step 1: Check system Python version and decide on installation
PYTHON_CMD=""
log_message "Searching for suitable Python (>= 3.11) in PATH..."

# Find all python3* executables in PATH and select the highest version >= 3.11
HIGHEST_VERSION=""
HIGHEST_CMD=""
PYTHON_CANDIDATES=$(compgen -c | grep -E '^python3(\.[0-9]+)?$' | sort -u)
for candidate in $PYTHON_CANDIDATES; do
    if command_exists "$candidate"; then
        VERSION=$($candidate --version 2>&1 | awk '{print $2}')
        MAJOR=$(echo "$VERSION" | cut -d. -f1)
        MINOR=$(echo "$VERSION" | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 11 ]; then
            # Compare versions to select the highest
            if [[ -z "$HIGHEST_VERSION" ]] || [[ $(printf '%s\n' "$VERSION" "$HIGHEST_VERSION" | sort -V | tail -n1) == "$VERSION" ]]; then
                HIGHEST_VERSION="$VERSION"
                HIGHEST_CMD="$candidate"
            fi
        fi
    fi
done

if [ -n "$HIGHEST_CMD" ]; then
    log_message "Found suitable Python: $HIGHEST_CMD ($HIGHEST_VERSION)"
    PYTHON_CMD="$HIGHEST_CMD"
fi

if [ -z "$PYTHON_CMD" ]; then
    handle_error 1 "No suitable Python (>= 3.11) found in PATH. Please install Python 3.11 or higher."
fi

log_message "Using Python at: $(which $PYTHON_CMD)"
log_message "Python setup complete."

# Step 2: Install all required packages at once
log_message "Installing required packages..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update && sudo apt-get install -y \
        pigz \
        cmake \
        libcurl4-openssl-dev \
        python3-venv \
        python3-pip \
        build-essential \
        git \
        ninja-build \
        nvidia-cuda-toolkit
elif command -v yum &> /dev/null; then
    sudo yum install -y pigz cmake libcurl-openssl-dev python3-venv python3-pip
elif command -v dnf &> /dev/null; then
    sudo dnf install -y pigz cmake libcurl-openssl-dev python3-venv python3-pip
else
    log_error "No supported package manager found (apt-get, yum, or dnf)"
    exit 1
fi
log_message "All required packages installed successfully"

# Step 3: Check for GPU offloading setup
log_message "Checking for GPU offloading setup..."
if command_exists prime-run; then
    log_message "NVIDIA Prime setup detected (prime-run available)"
    PRIME_RUN_AVAILABLE=true
elif command_exists optirun; then
    log_message "Optimus setup detected (optirun available)"
    OPTIRUN_AVAILABLE=true
elif command_exists primusrun; then
    log_message "Optimus setup detected (primusrun available)"
    PRIMUSRUN_AVAILABLE=true
else
    log_message "No GPU offloading setup detected, using direct GPU access"
    PRIME_RUN_AVAILABLE=false
    OPTIRUN_AVAILABLE=false
    PRIMUSRUN_AVAILABLE=false
fi

# Step 4: Pull llama-server cuda image
log_message "Pulling llama-server cuda image..."
if ! docker pull ghcr.io/ggerganov/llama.cpp:server-cuda; then
    handle_error 1 "Failed to pull llama.cpp server Docker image"
fi

# Step 5: Create llama-server wrapper script
log_message "Creating llama-server wrapper script..."
LLAMA_WRAPPER_DIR="$HOME/.local/bin"
mkdir -p "$LLAMA_WRAPPER_DIR"

# After venv creation and activation
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
VENV_SITE_PACKAGES="$(pwd)/local_ai/lib/${PYTHON_VERSION}/site-packages"
TEMPLATES_DIR="$VENV_SITE_PACKAGES/local_ai/examples/templates"

# Get the user's home directory for the absolute path expected by the Python code
USER_HOME="$HOME"
ABS_TEMPLATE_PATH="$USER_HOME/.local/lib/${PYTHON_VERSION}/site-packages/local_ai/examples/templates"

cat > "$LLAMA_WRAPPER_DIR/llama-server" << EOF
#!/bin/bash
# Wrapper script to run llama-server using Docker

MODELS_DIR="$(pwd)/llms-storage"
TEMPLATES_DIR="$TEMPLATES_DIR"

# Default port
PORT=11434
prev=""
for arg in "\$@"; do
    if [[ "\$prev" == "--port" ]]; then
        PORT="\$arg"
    fi
    prev="\$arg"
done

# Check for GPU offloading setup
if command -v prime-run >/dev/null 2>&1; then
    echo "Info: NVIDIA Prime setup detected, using prime-run for GPU access"
    GPU_ACCESS="prime-run"
elif command -v optirun >/dev/null 2>&1; then
    echo "Info: Optimus setup detected, using optirun for GPU access"
    GPU_ACCESS="optirun"
elif command -v primusrun >/dev/null 2>&1; then
    echo "Info: Optimus setup detected, using primusrun for GPU access"
    GPU_ACCESS="primusrun"
else
    GPU_ACCESS="direct"
fi

# Mount model and template dirs at all needed paths
MODEL_MOUNT="-v \$MODELS_DIR:/models -v \$MODELS_DIR:\$MODELS_DIR"
TEMPLATE_MOUNT="-v \$TEMPLATES_DIR:/templates -v \$TEMPLATES_DIR:\$TEMPLATES_DIR -v \$TEMPLATES_DIR:$ABS_TEMPLATE_PATH"

DOCKER_RUN="docker run --rm -it --gpus all -p \$PORT:\$PORT \$MODEL_MOUNT \$TEMPLATE_MOUNT ghcr.io/ggerganov/llama.cpp:server-cuda \$@"

if [ "\$GPU_ACCESS" = "prime-run" ]; then
    prime-run \$DOCKER_RUN
elif [ "\$GPU_ACCESS" = "optirun" ]; then
    optirun \$DOCKER_RUN
elif [ "\$GPU_ACCESS" = "primusrun" ]; then
    primusrun \$DOCKER_RUN
else
    \$DOCKER_RUN
fi
EOF

chmod +x "$LLAMA_WRAPPER_DIR/llama-server"
log_message "Created llama-server wrapper at $LLAMA_WRAPPER_DIR/llama-server"

# Add to PATH if not already there
if [[ ":$PATH:" != *":$LLAMA_WRAPPER_DIR:"* ]]; then
    log_message "Adding $LLAMA_WRAPPER_DIR to PATH..."
    echo "export PATH=\"$LLAMA_WRAPPER_DIR:\$PATH\"" >> ~/.bashrc
    export PATH="$LLAMA_WRAPPER_DIR:$PATH"
    log_message "PATH updated for current session and future sessions"
fi

# Verify llama-server is available
if ! command -v llama-server &> /dev/null; then
    handle_error 1 "Failed to make llama-server available in PATH"
fi
log_message "llama-server command is now available"

# Step 6: Create and activate virtual environment
log_message "Creating virtual environment 'local_ai'..."
"$PYTHON_CMD" -m venv local_ai || handle_error $? "Failed to create virtual environment"

log_message "Activating virtual environment..."
source local_ai/bin/activate || handle_error $? "Failed to activate virtual environment"
log_message "Virtual environment activated."

# Step 8: Install local-ai toolkit (with version check and update logic)
log_message "Setting up local-ai toolkit..."
if pip show local-ai &>/dev/null; then
    log_message "local-ai is installed. Checking for updates..."
    # Get installed version
    INSTALLED_VERSION=$(pip show local-ai | grep Version | awk '{print $2}')
    log_message "Current version: $INSTALLED_VERSION"
    # Get remote version (from GitHub repository without installing)
    log_message "Checking latest version from repository..."
    TEMP_VERSION_FILE=$(mktemp)
    if curl -s https://raw.githubusercontent.com/eternalai-org/local-ai/main/local_ai/__init__.py | grep -o "__version__ = \"[0-9.]*\"" | cut -d'"' -f2 > "$TEMP_VERSION_FILE"; then
        REMOTE_VERSION=$(cat "$TEMP_VERSION_FILE")
        rm "$TEMP_VERSION_FILE"
        log_message "Latest version: $REMOTE_VERSION"
        # Compare versions
        if [ "$(printf '%s\n' "$INSTALLED_VERSION" "$REMOTE_VERSION" | sort -V | head -n1)" = "$INSTALLED_VERSION" ] && [ "$INSTALLED_VERSION" != "$REMOTE_VERSION" ]; then
            log_message "New version available. Updating..."
            pip uninstall --break-system-packages local-ai -y || handle_error $? "Failed to uninstall local-ai"
            pip install --break-system-packages -q git+https://github.com/eternalai-org/local-ai.git || handle_error $? "Failed to update local-ai toolkit"
            log_message "local-ai toolkit updated to version $REMOTE_VERSION."
        else
            log_message "Already running the latest version. No update needed."
        fi
    else
        log_message "Could not check latest version. Proceeding with update to be safe..."
        pip uninstall --break-system-packages local-ai -y || handle_error $? "Failed to uninstall local-ai"
        pip install --break-system-packages -q git+https://github.com/eternalai-org/local-ai.git || handle_error $? "Failed to update local-ai toolkit"
        log_message "local-ai toolkit updated."
    fi
else
    log_message "Installing local-ai toolkit..."
    pip install --break-system-packages -q git+https://github.com/eternalai-org/local-ai.git || handle_error $? "Failed to install local-ai toolkit"
    log_message "local-ai toolkit installed."
fi

log_message "Setup completed successfully."
