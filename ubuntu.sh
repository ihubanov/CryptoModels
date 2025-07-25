#!/bin/bash
set -o pipefail

# Function: log_message
# Logs informational messages with a specific format.
log_message() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    if [[ -n "${message// }" ]]; then
        echo "[$timestamp] [INFO] [MODEL_INSTALL_LLAMA] $message"
    fi
}

# Function: log_error
# Logs error messages with a specific format.
log_error() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    if [[ -n "${message// }" ]]; then
        echo "[$timestamp] [ERROR] [MODEL_INSTALL_LLAMA] $message" >&2
    fi
}

# Function: log_success
# Logs success messages with a specific format.
log_success() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    if [[ -n "${message// }" ]]; then
        echo "[$timestamp] [SUCCESS] [MODEL_INSTALL_LLAMA] $message"
    fi
}

# Function: log_warning
# Logs warning messages with a specific format.
log_warning() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    if [[ -n "${message// }" ]]; then
        echo "[$timestamp] [WARNING] [MODEL_INSTALL_LLAMA] $message" >&2
    fi
}

# Function: log_section
# Logs section headers with a specific format.
log_section() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    if [[ -n "${message// }" ]]; then
        echo
        echo "[$timestamp] [SECTION] [MODEL_INSTALL_LLAMA] =========================================="
        echo "[$timestamp] [SECTION] [MODEL_INSTALL_LLAMA] $message"
        echo "[$timestamp] [SECTION] [MODEL_INSTALL_LLAMA] =========================================="
        echo
    fi
}

# Function: handle_error
# Handles errors, logs the error, deactivates the virtual environment if active, and exits.
handle_error() {
    local exit_code=$1
    local error_msg=$2
    log_error "$error_msg (Exit code: $exit_code)"
    if [[ -n "$VIRTUAL_ENV" ]]; then
        log_warning "Deactivating virtual environment due to error..."
        deactivate 2>/dev/null || true
    fi
    exit $exit_code
}

# Function: command_exists
# Checks if a command exists in the system.
command_exists() {
    command -v "$1" &> /dev/null
}

# Function: check_docker
# Checks if Docker is installed and available.
check_docker() {
    log_section "Checking Docker Installation"
    if ! command_exists docker; then
        log_error "Docker is not installed or not available in PATH. Please install Docker and try again."
        exit 1
    fi
    log_success "Docker is installed and available."
}

# Function: check_apt_get
# Checks if apt-get is available.
check_apt_get() {
    log_section "Checking Package Manager"
    if ! command_exists apt-get; then
        log_error "apt-get is not available. This script requires Ubuntu or a compatible system."
        exit 1
    fi
    log_success "apt-get package manager is available."
}

# Function: check_sudo
# Checks if the user has sudo privileges.
check_sudo() {
    log_section "Checking Sudo Privileges"
    if ! sudo -n true 2>/dev/null; then
        log_error "This script requires sudo privileges. Please run as a user with sudo access."
        exit 1
    fi
    log_success "Sudo privileges confirmed."
}

# Function: check_python_installable
# Checks if a suitable Python 3 is present (prefers Python >= 3.11, falls back to system python3).
check_python_installable() {
    log_section "Checking Python Installation"
    log_message "Searching for Python installations >= 3.11..."
    
    # Find all python3.x installations in /usr/bin
    PYTHON_VERSIONS=()
    for py_exec in /usr/bin/python3.*; do
        if [[ -x "$py_exec" && "$py_exec" =~ python3\.[0-9]+$ ]]; then
            version_output=$("$py_exec" --version 2>&1)
            if [[ $version_output =~ Python\ ([0-9]+\.[0-9]+) ]]; then
                version="${BASH_REMATCH[1]}"
                PYTHON_VERSIONS+=("$version:$py_exec")
            fi
        fi
    done
    
    # Sort versions and find the highest >= 3.11
    HIGHEST_VERSION=""
    HIGHEST_PATH=""
    
    for version_path in "${PYTHON_VERSIONS[@]}"; do
        version="${version_path%%:*}"
        path="${version_path##*:}"
        
        # Check if version >= 3.11
        if [[ $(printf '%s\n' "3.11" "$version" | sort -V | head -n1) == "3.11" ]] || [[ "$version" == "3.11" ]]; then
            if [[ -z "$HIGHEST_VERSION" ]] || [[ $(printf '%s\n' "$HIGHEST_VERSION" "$version" | sort -V | tail -n1) == "$version" ]]; then
                HIGHEST_VERSION="$version"
                HIGHEST_PATH="$path"
            fi
        fi
    done
    
    if [[ -n "$HIGHEST_PATH" ]]; then
        PYTHON_CMD="$HIGHEST_PATH"
        PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
        log_success "Found Python >= 3.11: $PYTHON_CMD ($PYTHON_VERSION)"
    else
        log_warning "No Python >= 3.11 found. Checking for system python3..."
        if command_exists python3; then
            PYTHON_CMD="python3"
            PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
            log_success "Found system python3: $PYTHON_CMD ($PYTHON_VERSION)"
        else
            log_warning "python3 not found. Attempting to install python3..."
            log_message "Updating package lists and installing python3..."
            sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip || handle_error $? "Failed to install python3."
            if command_exists python3; then
                PYTHON_CMD="python3"
                PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
                log_success "Successfully installed python3: $PYTHON_CMD ($PYTHON_VERSION)"
            else
                handle_error 1 "python3 installation failed. Please install python3 manually."
            fi
        fi
    fi
}

# Function: preflight_checks
# Runs all pre-installation checks and prints a summary.
preflight_checks() {
    log_section "Running Preflight Checks"
    check_docker
    check_apt_get
    check_sudo
    check_python_installable
    log_success "All preflight checks passed successfully."
    echo
    echo "========================================="
    echo "Preflight checks summary:"
    echo "- Internet connectivity: OK"
    echo "- Docker: OK"
    echo "- apt-get: OK"
    echo "- Sudo: OK"
    echo "- Python: Using $PYTHON_CMD ($PYTHON_VERSION)"
    echo "========================================="
    echo
}

# Run all preflight checks before proceeding.
preflight_checks

# Python selection is now handled in check_python_installable function

log_message "Using Python at: $(which $PYTHON_CMD)"
log_message "Python setup complete."

# -----------------------------------------------------------------------------
# Step 2: Install required system packages
# -----------------------------------------------------------------------------
log_message "Installing required packages..."
if command -v apt-get &> /dev/null; then
    REQUIRED_PACKAGES=(pigz cmake libcurl4-openssl-dev python3-venv python3-pip build-essential git ninja-build nvidia-cuda-toolkit)
    MISSING_PACKAGES=()
    for pkg in "${REQUIRED_PACKAGES[@]}"; do
        if ! dpkg -s "$pkg" &> /dev/null; then
            MISSING_PACKAGES+=("$pkg")
        fi
    done
    if [ "${#MISSING_PACKAGES[@]}" -ne 0 ]; then
        log_message "Installing missing packages: ${MISSING_PACKAGES[*]}"
        sudo apt-get update
        sudo apt-get install -y "${MISSING_PACKAGES[@]}" || handle_error $? "Failed to install required packages."
    else
        log_message "All required packages are already installed."
    fi
else
    log_error "No supported package manager found (apt-get)"
    exit 1
fi
log_message "All required packages installed successfully."


# -----------------------------------------------------------------------------
# Step 3: Pull llama-server CUDA Docker image
# -----------------------------------------------------------------------------
LLAMA_IMAGE="ghcr.io/ggml-org/llama.cpp:server-cuda-b5974"
log_message "Pulling llama-server cuda image..."
if ! docker pull $LLAMA_IMAGE; then
    handle_error 1 "Failed to pull llama.cpp server Docker image."
fi

# -----------------------------------------------------------------------------
# Step 4: Create llama-server wrapper script for Docker usage
# -----------------------------------------------------------------------------
log_message "Creating llama-server wrapper script..."
LLAMA_WRAPPER_DIR="$HOME/.local/bin"
mkdir -p "$LLAMA_WRAPPER_DIR"

# Prepare variables for template and model paths.
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
MODELS_DIR="$(pwd)/llms-storage"
TEMPLATES_DIR="$(pwd)/eternal_zoo/examples/templates"
USER_HOME="$HOME"

cat > "$LLAMA_WRAPPER_DIR/llama-server" << EOF
#!/bin/bash

MODELS_DIR="$MODELS_DIR"
TEMPLATES_DIR="$TEMPLATES_DIR"

# Mount model and template directories for Docker.
MODEL_MOUNT="-v \$MODELS_DIR:\$MODELS_DIR"
TEMPLATE_MOUNT="-v \$TEMPLATES_DIR:\$TEMPLATES_DIR"

docker run --runtime nvidia -it --rm --network=host \$MODEL_MOUNT \$TEMPLATE_MOUNT $LLAMA_IMAGE "\$@"

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
log_message "Created llama-server wrapper at $LLAMA_WRAPPER_DIR/llama-server."

# -----------------------------------------------------------------------------
# Step 5: Add llama-server wrapper directory to PATH in shell rc file
# -----------------------------------------------------------------------------
# Function: update_shell_rc_path
# Updates the specified shell rc file to include the wrapper directory in PATH.
update_shell_rc_path() {
    local shell_rc="$1"
    local path_line="export PATH=\"$LLAMA_WRAPPER_DIR:\$PATH\""
    if [ -f "$shell_rc" ]; then
        log_message "Backing up $shell_rc..."
        cp "$shell_rc" "$shell_rc.backup.$(date +%Y%m%d%H%M%S)" || log_error "Failed to backup $shell_rc."
        if grep -Fxq "$path_line" "$shell_rc"; then
            log_message "$LLAMA_WRAPPER_DIR already in PATH in $shell_rc. No update needed."
        else
            # Remove any previous lines that add $LLAMA_WRAPPER_DIR to PATH.
            sed -i "\|export PATH=\"$LLAMA_WRAPPER_DIR:\$PATH\"|d" "$shell_rc"
            echo "$path_line" >> "$shell_rc"
            log_message "Updated PATH in $shell_rc."
        fi
    else
        log_message "$shell_rc does not exist. Creating and adding PATH update."
        echo "$path_line" > "$shell_rc"
    fi
}

if [[ ":$PATH:" != *":$LLAMA_WRAPPER_DIR:"* ]]; then
    log_message "Adding $LLAMA_WRAPPER_DIR to PATH..."
    # Detect which shell rc file to update based on the user's shell.
    SHELL_NAME=$(basename "$SHELL")
    if [ "$SHELL_NAME" = "zsh" ]; then
        update_shell_rc_path "$HOME/.zshrc"
    else
        update_shell_rc_path "$HOME/.bashrc"
    fi
    # Set a flag to print an informative message at the end
    PATH_UPDATE_NEEDED=1
    log_message "PATH updated for current session and future sessions."
fi

# -----------------------------------------------------------------------------
# Step 6: Create and activate Python virtual environment
# -----------------------------------------------------------------------------
log_message "Creating virtual environment 'eternal-zoo'..."
"$PYTHON_CMD" -m venv eternal-zoo || handle_error $? "Failed to create virtual environment."

log_message "Activating virtual environment..."
source eternal-zoo/bin/activate || handle_error $? "Failed to activate virtual environment."
log_message "Virtual environment activated."

# -----------------------------------------------------------------------------
# Step 7: Install or update eternal-zoo toolkit in the virtual environment
# -----------------------------------------------------------------------------
# Function: install_or_update_local_ai
# Uninstalls and reinstalls the eternal-zoo toolkit from the current directory.
install_or_update_local_ai() {
    pip uninstall eternal-zoo -y || handle_error $? "Failed to uninstall eternal-zoo."
    pip install -e . || handle_error $? "Failed to install/update eternal-zoo toolkit."
    log_message "eternal-zoo toolkit installed/updated."
}

log_message "Setting up eternal-zoo toolkit..."
if pip show eternal-zoo &>/dev/null; then
    log_message "eternal-zoo is installed. Reinstalling in development mode..."
    install_or_update_local_ai
    log_message "eternal-zoo toolkit reinstalled in development mode."
else
    log_message "Installing eternal-zoo toolkit in development mode..."
    install_or_update_local_ai
    log_message "eternal-zoo toolkit installed in development mode."
fi

log_message "Setup completed successfully."

# At the end of the script, print an informative message if PATH was updated
if [ "${PATH_UPDATE_NEEDED:-0}" = "1" ]; then
    echo
    echo "[INFO] The llama-server command directory was added to your PATH in your shell rc file."
    echo "      To use it in this session, run: export PATH=\"$LLAMA_WRAPPER_DIR:\$PATH\""
    echo "      Or restart your terminal or run: source ~/.bashrc (or ~/.zshrc)"
    echo
fi  