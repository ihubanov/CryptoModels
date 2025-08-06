from pathlib import Path

DEFAULT_MODEL_DIR = Path.home() / ".eternal-zoo" / "models"
POSTFIX_MODEL_PATH = ".gguf"
GATEWAY_URLS = [
    # Lighthouse's own gateway (recommended for files stored on Lighthouse)
    "https://gateway.lighthouse.storage/ipfs/",
    # Mesh3 public gateway
    "https://gateway.mesh3.network/ipfs/",
    # IPFS official gateway
    "https://ipfs.io/ipfs/",
    # Cloudflare IPFS gateway
    # "https://cloudflare-ipfs.com/ipfs/",
    # Cloudflare alternative
    # "https://cf-ipfs.com/ipfs/",
    # Pinata-backed gateway
    "https://dweb.link/ipfs/",
    # 4everland Filecoin ecosystem gateway
    # "https://4everland.io/ipfs/",
    # Infura IPFS gateway
    "https://infura-ipfs.io/ipfs/",
]

ETERNAL_AI_METADATA_GW = "https://agent.api.eternalai.org/api/vibe/model/"
PREFIX_DOWNLOAD_LOG = "[CRYPTOAGENTS_LOGGER] [MODEL_INSTALL]"
NOT_SUPPORTED_TOOLS_MODELS = ["gpt-oss-20b", "gpt-oss-120b", "bafkreiclhnqcjfbbusqmg73jcwasomv7tqchkqm3fea5wwzs5vavc2wzfq"]
