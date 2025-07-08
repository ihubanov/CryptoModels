.PHONY: download

# Default Filecoin/IPFS hash (change as needed)
HASH=bafkreiclwlxc56ppozipczuwkmgnlrxrerrvaubc5uhvfs3g2hp3lftrwm

# Download model using the specified hash
# Usage:
#   make download            # uses the default HASH
#   make download HASH=your_filecoin_hash_here

download:
	python crypto_models/download.py $(HASH) 