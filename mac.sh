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

# Version comparison function
compare_versions() {
    local installed="$1"
    local remote="$2"
    
    # Validate version strings (basic semver check)
    if [[ ! "$installed" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?([.-][a-zA-Z0-9]+)*$ ]]; then
        log_error "Invalid installed version format: $installed"
        return 2
    fi
    
    if [[ ! "$remote" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?([.-][a-zA-Z0-9]+)*$ ]]; then
        log_error "Invalid remote version format: $remote"
        return 2
    fi
    
    # Return 0 if update needed (remote > installed), 1 if no update needed, 2 if error
    if [ "$installed" != "$remote" ] && [ "$(printf '%s\n' "$installed" "$remote" | sort -V | head -n1)" = "$installed" ]; then
        return 0  # Update needed
    else
        return 1  # No update needed
    fi
}

# Generic package update function
update_package() {
    local package_name="$1"
    local github_url="$2"
    local version_source_url="$3"
    local version_regex="$4"
    local install_cmd="$5"
    
    log_message "Checking $package_name installation..."
    
    if pip show "$package_name" &>/dev/null; then
        log_message "$package_name is installed. Checking for updates..."
        
        # Get installed version
        local installed_version=$(pip show "$package_name" | grep Version | awk '{print $2}')
        log_message "Current $package_name version: $installed_version"
        
        # Get remote version
        log_message "Checking latest version from repository..."
        local temp_file=$(mktemp)
        local remote_version=""
        
        if curl -s --connect-timeout 10 --max-time 30 "$version_source_url" | grep -o "$version_regex" | cut -d'"' -f2 > "$temp_file" 2>/dev/null; then
            remote_version=$(cat "$temp_file")
            rm -f "$temp_file"
            
            if [ -n "$remote_version" ]; then
                log_message "Latest $package_name version: $remote_version"
                
                # Compare versions using our function
                if compare_versions "$installed_version" "$remote_version"; then
                    log_message "New version available. Updating $package_name..."
                    if pip uninstall "$package_name" -y && eval "$install_cmd"; then
                        log_message "$package_name updated to version $remote_version."
                    else
                        log_error "Failed to update $package_name. Continuing with installation..."
                    fi
                else
                    case $? in
                        1) log_message "Already running the latest version of $package_name. No update needed." ;;
                        2) log_error "Version comparison failed for $package_name. Skipping update." ;;
                    esac
                fi
            else
                log_message "Could not determine latest version. Skipping update for safety."
            fi
        else
            rm -f "$temp_file"
            log_message "Could not check latest version from repository. Skipping update for safety."
        fi
    else
        log_message "Installing $package_name..."
        if eval "$install_cmd"; then
            log_message "$package_name installed successfully."
        else
            log_error "Failed to install $package_name. Continuing with installation..."
        fi
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

export PATH="/opt/homebrew/bin/:$PATH"
export PATH="$HOME/homebrew/bin:$PATH"

# Step 1: Ensure Homebrew is installed and set PATH
if ! command_exists brew; then
    log_error "Homebrew is not installed. Please install Homebrew first: https://brew.sh/"
    exit 1
fi

BREW_PREFIX=$(brew --prefix)
export PATH="$BREW_PREFIX/bin:$PATH"
log_message "Homebrew found at $BREW_PREFIX. PATH updated for this session."

# Step 2: Check system Python version and decide on installation
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

# Step 3: Update PATH in .zshrc for future sessions
log_message "Checking if PATH update is needed in .zshrc..."
if ! grep -q "export PATH=\"/usr/local/bin:\$PATH\"" ~/.zshrc 2>/dev/null; then
    if [ -f ~/.zshrc ]; then
        log_message "Backing up current .zshrc..."
        cp ~/.zshrc ~/.zshrc.backup.$(date +%Y%m%d%H%M%S) || handle_error $? "Failed to backup .zshrc"
    else
        log_message "No existing .zshrc found. Skipping backup."
    fi
    
    log_message "Updating PATH in .zshrc for brew at /usr/local/bin..."
    echo "export PATH=\"/usr/local/bin:\$PATH\"" >> ~/.zshrc || handle_error $? "Failed to update .zshrc"
    log_message "Please restart your terminal or run 'source ~/.zshrc' for future sessions."
else
    log_message "PATH already updated in .zshrc."
fi

# Step 4: Install pigz
log_message "Checking for pigz installation..."
if command_exists pigz; then
    log_message "pigz is already installed."
else
    log_message "Installing pigz..."
    brew install pigz || handle_error $? "Failed to install pigz"
    log_message "pigz installed successfully."
fi

# Step 5: Install llama.cpp
log_message "Checking llama.cpp version..."
if command_exists llama-cli; then
    INSTALLED_VERSION=$(llama-cli --version 2>&1 | grep -oE 'version: [0-9]+' | cut -d' ' -f2 || echo "0")
    log_message "Current llama.cpp version: $INSTALLED_VERSION"
    
    # Get latest version from the formula file
    LATEST_VERSION=$(grep -oE 'tag: *"b[0-9]+"' llama.cpp.rb | sed 's/tag: *"b//;s/"//' || echo "0")
    log_message "Latest available version: $LATEST_VERSION"
    
    if [ "$INSTALLED_VERSION" -ne "$LATEST_VERSION" ]; then
        log_message "Version mismatch detected. Reinstalling llama.cpp to match formula version..."
        brew uninstall --force llama.cpp
        brew install llama.cpp.rb || handle_error $? "Failed to install llama.cpp"
    else
        log_message "Already running the correct version of llama.cpp."
    fi
else
    log_message "llama.cpp not found. Installing..."
    brew install llama.cpp.rb || handle_error $? "Failed to install llama.cpp"
fi

log_message "Verifying llama.cpp version..."
hash -r
llama-cli --version || handle_error $? "llama.cpp verification failed"
log_message "llama.cpp setup complete."


# Step 6: Create and activate virtual environment
log_message "Creating virtual environment 'cryptomodels'..."
"$PYTHON_CMD" -m venv cryptomodels || handle_error $? "Failed to create virtual environment"

log_message "Activating virtual environment..."
source cryptomodels/bin/activate || handle_error $? "Failed to activate virtual environment"
log_message "Virtual environment activated."


# Step 7: Install mflux dependencies
log_message "Checking mlx-flux installation..."
update_package "mlx-flux" "https://github.com/0x9334/mlx-flux.git" "https://raw.githubusercontent.com/0x9334/mlx-flux/main/setup.py" "version=\"[0-9.]*\"" "pip install git+https://github.com/0x9334/mlx-flux.git"
PYTHON_VERSION_MAJOR_MINOR=$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "$PYTHON_VERSION_MAJOR_MINOR" = "3.13" ]; then
    log_message "Detected Python 3.13. Installing compatible SentencePiece wheel..."
    if pip install -q https://github.com/anthonywu/sentencepiece/releases/download/0.2.1-py13dev/sentencepiece-0.2.1-cp313-cp313-macosx_11_0_arm64.whl; then
        log_message "SentencePiece for Python 3.13 installed successfully."
    else
        log_error "Failed to install SentencePiece for Python 3.13. Continuing with installation..."
    fi
fi

# Step 8: Install cryptomodels toolkit
log_message "Setting up cryptomodels toolkit..."
update_package "cryptomodels" "https://github.com/eternalai-org/CryptoModels.git" "https://raw.githubusercontent.com/eternalai-org/CryptoModels/main/crypto_models/__init__.py" "__version__ = \"[0-9.]*\"" "pip install -q git+https://github.com/eternalai-org/CryptoModels.git"

log_message "Setup completed successfully."