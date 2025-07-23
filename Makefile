# Makefile for CryptoModels installation
# Sets specific versions for mlx-flux and cryptomodels packages

# Package versions
MLX_FLUX_TAG=1.0.5
CRYPTOMODELS_TAG=1.1.31

# Default target
.PHONY: install
install:
	@echo "Installing CryptoModels with specific versions:"
	@echo "  MLX_FLUX_TAG: $(MLX_FLUX_TAG)"
	@echo "  CRYPTOMODELS_TAG: $(CRYPTOMODELS_TAG)"
	@echo ""
	MLX_FLUX_TAG=$(MLX_FLUX_TAG) CRYPTOMODELS_TAG=$(CRYPTOMODELS_TAG) ./mac.sh

# Clean target to remove virtual environment
.PHONY: clean
clean:
	@echo "Cleaning up virtual environment..."
	rm -rf cryptomodels/
	@echo "Cleanup complete."

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  install  - Install CryptoModels with specific package versions"
	@echo "  clean    - Remove the virtual environment"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Package versions:"
	@echo "  MLX_FLUX_TAG: $(MLX_FLUX_TAG)"
	@echo "  CRYPTOMODELS_TAG: $(CRYPTOMODELS_TAG)" 