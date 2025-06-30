"""
Configuration management for CryptoModels.

This module provides centralized configuration management with support for:
- Environment variables
- Default values
- Type validation
- Easy customization for different environments
"""

import os
from typing import Optional, Union
from pathlib import Path


class PerformanceConfig:
    """Performance and timeout configuration settings."""
    
    # Dynamic unload feature settings
    IDLE_TIMEOUT: int = int(os.getenv("CRYPTO_IDLE_TIMEOUT", "600"))  # 10 minutes
    UNLOAD_CHECK_INTERVAL: int = int(os.getenv("CRYPTO_UNLOAD_CHECK_INTERVAL", "30"))  # 30 seconds
    
    # Service timeouts
    SERVICE_START_TIMEOUT: int = int(os.getenv("CRYPTO_SERVICE_START_TIMEOUT", "120"))  # 2 minutes
    HTTP_TIMEOUT: float = float(os.getenv("CRYPTO_HTTP_TIMEOUT", "180.0"))  # 3 minutes
    STREAM_TIMEOUT: float = float(os.getenv("CRYPTO_STREAM_TIMEOUT", "300.0"))  # 5 minutes
    
    # HTTP connection pooling
    POOL_CONNECTIONS: int = int(os.getenv("CRYPTO_POOL_CONNECTIONS", "50"))
    POOL_KEEPALIVE: int = int(os.getenv("CRYPTO_POOL_KEEPALIVE", "10"))
    
    # Retry and error handling
    MAX_RETRIES: int = int(os.getenv("CRYPTO_MAX_RETRIES", "2"))
    RETRY_DELAY: float = float(os.getenv("CRYPTO_RETRY_DELAY", "0.5"))
    
    # Queue and processing
    MAX_QUEUE_SIZE: int = int(os.getenv("CRYPTO_MAX_QUEUE_SIZE", "50"))
    HEALTH_CHECK_INTERVAL: int = int(os.getenv("CRYPTO_HEALTH_CHECK_INTERVAL", "2"))
    
    # Streaming
    STREAM_CHUNK_SIZE: int = int(os.getenv("CRYPTO_STREAM_CHUNK_SIZE", "16384"))  # 16KB


class CoreConfig:
    """Core service configuration settings."""
    
    # Lock and process management timeouts
    LOCK_TIMEOUT: int = int(os.getenv("CRYPTO_LOCK_TIMEOUT", "1800"))  # 30 minutes
    PORT_CHECK_TIMEOUT: int = int(os.getenv("CRYPTO_PORT_CHECK_TIMEOUT", "2"))
    HEALTH_CHECK_TIMEOUT: int = int(os.getenv("CRYPTO_HEALTH_CHECK_TIMEOUT", "300"))  # 5 minutes
    PROCESS_TERM_TIMEOUT: int = int(os.getenv("CRYPTO_PROCESS_TERM_TIMEOUT", "15"))
    MAX_PORT_RETRIES: int = int(os.getenv("CRYPTO_MAX_PORT_RETRIES", "5"))
    
    # HTTP request settings
    REQUEST_RETRIES: int = int(os.getenv("CRYPTO_REQUEST_RETRIES", "2"))
    REQUEST_DELAY: int = int(os.getenv("CRYPTO_REQUEST_DELAY", "2"))
    REQUEST_TIMEOUT: int = int(os.getenv("CRYPTO_REQUEST_TIMEOUT", "8"))


class ModelConfig:
    """Model-related configuration settings."""
    
    # Default model names
    DEFAULT_CHAT_MODEL: str = os.getenv("CRYPTO_DEFAULT_CHAT_MODEL", "default-chat-model")
    DEFAULT_EMBED_MODEL: str = os.getenv("CRYPTO_DEFAULT_EMBED_MODEL", "default-embed-model")
    
    # Model limits
    MAX_MESSAGES: int = int(os.getenv("CRYPTO_MAX_MESSAGES", "50"))
    DEFAULT_MAX_TOKENS: int = int(os.getenv("CRYPTO_DEFAULT_MAX_TOKENS", "8192"))
    DEFAULT_CONTEXT_LENGTH: int = int(os.getenv("CRYPTO_DEFAULT_CONTEXT_LENGTH", "32768"))


class FilePathConfig:
    """File path and directory configuration."""
    
    # Service files
    RUNNING_SERVICE_FILE: str = os.getenv("RUNNING_SERVICE_FILE", "running_service.msgpack")
    START_LOCK_FILE: str = os.getenv("START_LOCK_FILE", "start_lock.lock")
    
    # Directories
    LOGS_DIR: str = os.getenv("CRYPTO_LOGS_DIR", "logs")
    
    # External commands (these are set in __init__.py but can be overridden)
    LLAMA_SERVER: Optional[str] = os.getenv("LLAMA_SERVER")
    TAR_COMMAND: Optional[str] = os.getenv("TAR_COMMAND")
    PIGZ_COMMAND: Optional[str] = os.getenv("PIGZ_COMMAND")
    CAT_COMMAND: Optional[str] = os.getenv("CAT_COMMAND")


class NetworkConfig:
    """Network and server configuration."""

    # APIS
    DEFAULT_PORT: int = int(os.getenv("CRYPTO_DEFAULT_PORT", "8080"))
    DEFAULT_HOST: str = os.getenv("CRYPTO_DEFAULT_HOST", "0.0.0.0")
    
    # Download settings
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("CRYPTO_DEFAULT_CHUNK_SIZE", "8192"))
    MAX_CONCURRENT_DOWNLOADS: int = int(os.getenv("CRYPTO_MAX_CONCURRENT_DOWNLOADS", "8"))
    DOWNLOAD_TIMEOUT: int = int(os.getenv("CRYPTO_DOWNLOAD_TIMEOUT", "300"))  # 5 minutes


class Config:
    """Main configuration class that aggregates all config sections."""
    
    performance = PerformanceConfig()
    core = CoreConfig()
    model = ModelConfig()
    file_paths = FilePathConfig()
    network = NetworkConfig()
    
    @classmethod
    def get_env_summary(cls) -> dict:
        """Get a summary of all configuration values for debugging."""
        return {
            "performance": {
                attr: getattr(cls.performance, attr)
                for attr in dir(cls.performance)
                if not attr.startswith("_") and attr.isupper()
            },
            "core": {
                attr: getattr(cls.core, attr)
                for attr in dir(cls.core)
                if not attr.startswith("_") and attr.isupper()
            },
            "model": {
                attr: getattr(cls.model, attr)
                for attr in dir(cls.model)
                if not attr.startswith("_") and attr.isupper()
            },
            "file_paths": {
                attr: getattr(cls.file_paths, attr)
                for attr in dir(cls.file_paths)
                if not attr.startswith("_") and attr.isupper()
            },
            "network": {
                attr: getattr(cls.network, attr)
                for attr in dir(cls.network)
                if not attr.startswith("_") and attr.isupper()
            }
        }
    
    @classmethod
    def print_config(cls) -> None:
        """Print current configuration for debugging."""
        import json
        print("Current CryptoModels Configuration:")
        print(json.dumps(cls.get_env_summary(), indent=2, default=str))


# Create a global config instance
config = Config() 