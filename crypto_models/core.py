import os
import json
import time
import signal
import msgpack
import psutil
import asyncio
import socket
import requests
import subprocess
import pkg_resources
import shutil
from pathlib import Path
from loguru import logger
from crypto_models.config import config
from crypto_models.utils import wait_for_health
from typing import Optional, Dict, Any, List
from crypto_models.download import download_model_async, fetch_model_metadata_async
from crypto_models.constants import DEFAULT_MODEL_DIR, POSTFIX_MODEL_PATH


class CryptoAgentsServiceError(Exception):
    """Base exception for CryptoModels service errors."""
    pass

class ServiceStartError(CryptoAgentsServiceError):
    """Exception raised when service fails to start."""
    pass

class ModelNotFoundError(CryptoAgentsServiceError):
    """Exception raised when model file is not found."""
    pass

class CryptoModelsManager:
    """Manages a CryptoModels service with optimized performance."""
    
    def __init__(self):
        """Initialize the CryptoModelsManager with optimized defaults."""
        # Performance constants from config
        self.LOCK_TIMEOUT = config.core.LOCK_TIMEOUT
        self.PORT_CHECK_TIMEOUT = config.core.PORT_CHECK_TIMEOUT
        self.HEALTH_CHECK_TIMEOUT = config.core.HEALTH_CHECK_TIMEOUT
        self.PROCESS_TERM_TIMEOUT = config.core.PROCESS_TERM_TIMEOUT
        self.MAX_PORT_RETRIES = config.core.MAX_PORT_RETRIES
        
        # File paths from config
        self.msgpack_file = Path(config.file_paths.RUNNING_SERVICE_FILE)
        self.loaded_models: Dict[str, Any] = {}
        self.llama_server_path = config.file_paths.LLAMA_SERVER
        self.start_lock_file = Path(config.file_paths.START_LOCK_FILE)
        self.logs_dir = Path(config.file_paths.LOGS_DIR)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Clean up any stale temporary lock files on initialization
        self._cleanup_temp_lock_files()
        
    def _cleanup_temp_lock_files(self) -> None:
        """Clean up any temporary lock files left behind by previous runs."""
        try:
            # Look for .tmp files in the same directory as the lock file
            lock_dir = self.start_lock_file.parent
            
            for temp_file in lock_dir.glob(f"*.tmp"):
                # Only remove temp files that match our naming pattern
                if temp_file.name.startswith(self.start_lock_file.stem):
                    try:
                        temp_file.unlink()
                        logger.debug(f"Cleaned up stale temporary lock file: {temp_file}")
                    except OSError as e:
                        logger.warning(f"Failed to clean up temporary lock file {temp_file}: {e}")
        except Exception as e:
            logger.warning(f"Error during temp lock file cleanup: {e}")
            
    def _get_free_port(self) -> int:
        """Get a free port number."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def _get_family_template_and_practice(self, model_family: str):
        """Helper to get template and best practice paths based on folder name."""
        return (
            self._get_model_template_path(model_family),
            self._get_model_best_practice_path(model_family)
        )
    
    def _retry_request_json(self, url: str, retries: int = None, delay: int = None, timeout: int = None) -> Optional[dict]:
        """
        Utility to retry a GET request for JSON data with optimized parameters.
        Returns parsed JSON data or None on failure.
        """
        # Use config values as defaults
        retries = retries or config.core.REQUEST_RETRIES
        delay = delay or config.core.REQUEST_DELAY
        timeout = timeout or config.core.REQUEST_TIMEOUT
        
        backoff_delay = delay
        last_error = None
        
        for attempt in range(retries):
            try:
                # Use session for connection pooling if making multiple requests
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()  # Raise exception for HTTP errors
                
                # Parse JSON once and return
                return response.json()
                
            except requests.exceptions.Timeout as e:
                last_error = f"Timeout after {timeout}s"
                logger.warning(f"Attempt {attempt+1}/{retries} timed out for {url}")
            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {str(e)[:100]}"
                logger.warning(f"Attempt {attempt+1}/{retries} connection failed for {url}")
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP error {e.response.status_code}"
                logger.warning(f"Attempt {attempt+1}/{retries} HTTP error for {url}: {e}")
                # Don't retry on 4xx errors (client errors)
                if 400 <= e.response.status_code < 500:
                    break
            except (requests.exceptions.RequestException, ValueError) as e:
                last_error = f"Request error: {str(e)[:100]}"
                logger.warning(f"Attempt {attempt+1}/{retries} failed for {url}: {e}")
            
            # Sleep with exponential backoff (except on last attempt)
            if attempt < retries - 1:
                time.sleep(backoff_delay)
                backoff_delay = min(backoff_delay * 1.5, 8)  # Cap at 8 seconds
        
        logger.error(f"Failed to fetch {url} after {retries} attempts. Last error: {last_error}")
        return None

    def start(self, hashes: str, port: int = None, host: str = None, context_length: int = None) -> bool:
        """
        Start the CryptoModels service with multi-model support and on-demand loading.

        Args:
            hashes (str): Comma-separated string of model hashes. First hash is main model (loaded immediately),
                         subsequent hashes are stored for on-demand loading. Single hash also supported.
            port (int): Port number for the CryptoModels service (default from config).
            host (str): Host address for the CryptoModels service (default from config).
            context_length (int): Context length for the model (default from config).

        Returns:
            bool: True if service started successfully, False otherwise.

        Raises:
            ValueError: If no hashes are provided.
            ModelNotFoundError: If model file is not found.
            ServiceStartError: If service fails to start.
        """
        # Parse comma-separated hashes
        if not hashes or not hashes.strip():
            raise ValueError("At least one model hash is required to start the service")
        
        # Split by comma and clean whitespace
        hashes_list = [h.strip() for h in hashes.split(',') if h.strip()]
        
        if not hashes_list:
            raise ValueError("At least one model hash is required to start the service")
        
        # Use config defaults if not provided
        port = port or config.network.DEFAULT_PORT
        host = host or config.network.DEFAULT_HOST
        context_length = context_length or config.model.DEFAULT_CONTEXT_LENGTH

        # Main model is the first hash (on_demand: false)
        main_hash = hashes_list[0]
        on_demand_hashes = hashes_list[1:] if len(hashes_list) > 1 else []

        # Acquire process lock to prevent concurrent starts
        if not self._acquire_start_lock(main_hash, host, port):
            return False
        
        # Track processes for cleanup on interruption
        ai_process = None
        apis_process = None
        
        def cleanup_processes():
            """Clean up any running processes."""
            if ai_process and ai_process.poll() is None:
                logger.info("Cleaning up AI process due to interruption...")
                try:
                    ai_process.terminate()
                    ai_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    ai_process.kill()
                except Exception as e:
                    logger.error(f"Error cleaning up AI process: {e}")
            
            if apis_process and apis_process.poll() is None:
                logger.info("Cleaning up API process due to interruption...")
                try:
                    apis_process.terminate()
                    apis_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    apis_process.kill()
                except Exception as e:
                    logger.error(f"Error cleaning up API process: {e}")
        
        def signal_handler(signum, frame):
            """Handle interruption signals."""
            logger.info(f"Received signal {signum}, cleaning up processes...")
            cleanup_processes()
            self._release_start_lock()
            exit(1)
        
        # Set up signal handlers for graceful cleanup
        original_sigint = signal.signal(signal.SIGINT, signal_handler)
        original_sigterm = signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Check if the requested port is available before doing expensive operations
            if not self._check_port_availability(host, port):
                raise ServiceStartError(f"Port {port} is already in use on {host}")

            try:
                logger.info(f"Starting CryptoModels service with {len(hashes_list)} models")
                logger.info(f"Main model hash: {main_hash}")
                if on_demand_hashes:
                    logger.info(f"On-demand models: {on_demand_hashes}")
                
                # Download and prepare all models
                models_info = {}
                for i, hash_val in enumerate(hashes_list):
                    
                    logger.info(f"Downloading model {i+1}/{len(hashes_list)}: {hash_val}")
                    success, local_model_path = asyncio.run(download_model_async(hash_val))
                    if not success:
                        raise ModelNotFoundError(f"Model file not found for hash: {hash_val}")
                    
                    success, metadata = asyncio.run(fetch_model_metadata_async(hash_val))
                    
                    if not success:
                        raise ModelNotFoundError(f"Model metadata not found for hash: {hash_val}")
                    
                    metadata.pop("files")
                    
                    models_info[hash_val] = {
                        "local_model_path": local_model_path,
                        "metadata": metadata,
                        "on_demand": i > 0,  # First model is not on-demand
                        "context_length": context_length,
                    }
                    
                    is_lora = metadata.get("lora", False)
                    lora_config = None
                    if is_lora:
                        lora_config = {}
                        lora_metadata_path = os.path.join(local_model_path, "metadata.json")
                        lora_metadata, error_msg = self._load_lora_metadata(lora_metadata_path)
                        if lora_metadata is None:
                            logger.warning(f"Error loading LoRA metadata: {error_msg}")
                            continue
                        base_model_hash = lora_metadata["base_model"]
                        success, base_model_path = asyncio.run(download_model_async(base_model_hash))
                        if not success:
                            raise ModelNotFoundError(f"Base model file not found for hash: {base_model_hash}")
                        lora_scales = lora_metadata["lora_scales"]
                        if len(lora_metadata["lora_paths"]) == 0:
                            logger.warning(f"No LoRA paths found in metadata for hash: {hash_val}")
                            continue
                        for i, lora_path in enumerate(lora_metadata["lora_paths"]):
                            absolute_lora_path = os.path.join(local_model_path, lora_path)
                            lora_config[str(i)] = {
                                "path": absolute_lora_path,
                                "scale": lora_scales[i]
                            }
                        models_info[hash_val]["lora_config"] = lora_config
                        models_info[hash_val]["base_model_path"] = base_model_path

                    is_multimodal, projector_path = self._check_multimodal_support(local_model_path) 
                    models_info[hash_val]["multimodal"] = is_multimodal
                    models_info[hash_val]["local_projector_path"] = projector_path

                # Check if any existing model is running
                model_running = self.get_running_model()
                if model_running:
                    if model_running == main_hash:
                        logger.warning(f"Main model '{main_hash}' already running on port {port}")
                        return True
                    logger.info(f"Stopping existing model '{model_running}' on port {port}")
                    self.stop(force=True)

                # Start the main model
                main_model_info = models_info[main_hash]
                local_model_path = main_model_info["local_model_path"]
                metadata = main_model_info["metadata"]
                
                folder_name = metadata.get("folder_name", "")
                family = metadata.get("family", None)
                ram = metadata.get("ram", None)
                task = metadata.get("task", "chat")
                config_name = metadata.get("config_name", "flux-dev")
                is_lora = metadata.get("lora", False)
                local_ai_port = self._get_free_port()
                
                # Build command and service metadata for main model
                if task == "embed":
                    running_ai_command = self._build_embed_command(local_model_path, local_ai_port, host)
                    service_metadata = self._create_service_metadata(
                        main_hash, local_model_path, local_ai_port, port, context_length, task, False, None, None
                    )
                elif task == "image-generation":
                    if not shutil.which("mlx-flux"):
                        raise CryptoAgentsServiceError("mlx-flux command not found in PATH")
                    
                    effective_model_path = local_model_path
                    lora_paths = None
                    lora_scales = None
                    
                    if is_lora:
                        base_model_path = main_model_info["base_model_path"]
                        lora_config = main_model_info["lora_config"]

                        lora_paths = []
                        lora_scales = []
                        for i in range(len(lora_config)):
                            lora_paths.append(lora_config[str(i)]["path"])
                            lora_scales.append(lora_config[str(i)]["scale"])
                        
                        effective_model_path = base_model_path
                        
                        logger.info(f"LoRA model detected - using base model: {base_model_path}")
                        logger.info(f"LoRA paths: {lora_paths}")
                        logger.info(f"LoRA scales: {lora_scales}")
                                
                    running_ai_command = self._build_image_generation_command(
                        effective_model_path, local_ai_port, host, config_name, lora_paths, lora_scales
                    )
                    service_metadata = self._create_service_metadata(
                        main_hash, local_model_path, local_ai_port, port, context_length, task, False, None, lora_config
                    )
                else:
                    is_multimodal = main_model_info["multimodal"]
                    projector_path = main_model_info["local_projector_path"]
                    
                    service_metadata = self._create_service_metadata(
                        main_hash, local_model_path, local_ai_port, port, context_length, 
                        task, is_multimodal, projector_path
                    )

                    # Build command based on model family
                    running_ai_command = self._build_model_command(folder_name, local_model_path, local_ai_port, host, context_length)

                    if service_metadata["multimodal"]:
                        running_ai_command.extend([
                            "--mmproj", str(projector_path)
                        ])

                # Add main model metadata
                service_metadata["family"] = family
                service_metadata["folder_name"] = folder_name
                service_metadata["ram"] = ram
                service_metadata["running_ai_command"] = running_ai_command
                
                # Add multi-model information
                service_metadata["models"] = {}
                for hash_val, model_info in models_info.items():
                    service_metadata["models"][hash_val] = {
                        "local_model_path": model_info["local_model_path"],
                        "local_projector_path": model_info["local_projector_path"],
                        "metadata": model_info["metadata"],
                        "on_demand": model_info["on_demand"],
                        "active": hash_val == main_hash,
                        "lora_config": model_info.get("lora_config", None),
                        "base_model_path": model_info.get("base_model_path", None),
                        "context_length": model_info.get("context_length", 32768)
                    }
            
                logger.info(f"Starting main model process: {' '.join(running_ai_command)}")
                
                # Create log files for AI process
                ai_log_stderr = self.logs_dir / "ai.log"
                try:
                    with open(ai_log_stderr, 'w') as stderr_log:
                        ai_process = subprocess.Popen(
                            running_ai_command,
                            stderr=stderr_log,
                            preexec_fn=os.setsid
                        )
                    logger.info(f"AI logs written to {ai_log_stderr}")
                except Exception as e:
                    logger.error(f"Error starting CryptoModels service: {str(e)}", exc_info=True)
                    cleanup_processes()
                    return False
        
                if not wait_for_health(local_ai_port):
                    logger.error(f"Service failed to start within 600 seconds")
                    cleanup_processes()
                    return False
                
                logger.info(f"[CRYPTOMODELS] Main model service started on port {local_ai_port}")

                # Start the FastAPI app
                uvicorn_command = [
                    "uvicorn",
                    "crypto_models.apis:app",
                    "--host", host,
                    "--port", str(port),
                    "--log-level", "info"
                ]
                logger.info(f"Starting API process: {' '.join(uvicorn_command)}")
                
                api_log_stderr = self.logs_dir / "api.log"
                try:
                    with open(api_log_stderr, 'w') as stderr_log:
                        apis_process = subprocess.Popen(
                            uvicorn_command,
                            stderr=stderr_log,
                            preexec_fn=os.setsid
                        )
                    logger.info(f"API logs written to {api_log_stderr}")
                except Exception as e:
                    logger.error(f"Error starting FastAPI app: {str(e)}", exc_info=True)
                    cleanup_processes()
                    return False
                
                if not wait_for_health(port):
                    logger.error(f"API service failed to start within 600 seconds")
                    cleanup_processes()
                    return False

                logger.info(f"Multi-model service started on port {port}")
                if on_demand_hashes:
                    logger.info(f"On-demand models ready: {on_demand_hashes}")

                # Update service metadata with process IDs
                service_metadata.update({
                    "pid": ai_process.pid,
                    "app_pid": apis_process.pid
                })

                self._dump_running_service(service_metadata)

                # Update service metadata to the FastAPI app
                try:
                    update_url = f"http://localhost:{port}/update"
                    response = requests.post(update_url, json=service_metadata, timeout=10)
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to update service metadata: {str(e)}")
                    cleanup_processes()
                    return False
                
                return True

            except Exception as e:
                logger.error(f"Error starting CryptoModels service: {str(e)}", exc_info=True)
                cleanup_processes()
                return False
                
        finally:
            # Restore original signal handlers
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)
            # Always remove the lock when done (success or failure)
            self._release_start_lock()

    def _dump_running_service(self, metadata: dict):
        """Dump the running service details to a file."""
        try:
            with open(self.msgpack_file, "wb") as f:
                msgpack.dump(metadata, f)
                
        except Exception as e:
            logger.error(f"Error dumping running service: {str(e)}", exc_info=True)
            return False

    def get_running_model(self) -> Optional[str]:
        """
        Get currently running model hash if the service is healthy.

        Returns:
            Optional[str]: Running model hash or None if no healthy service exists.
        """
        if not self.msgpack_file.exists():
            return None

        try:
            # Load service info from msgpack file
            with open(self.msgpack_file, "rb") as f:
                service_info = msgpack.load(f)   
            model_hash = service_info.get("hash")      
            return model_hash

        except Exception as e:
            logger.error(f"Error getting running model: {str(e)}")
            return None
    
    def stop(self, force: bool = False) -> bool:
        """
        Stop the running CryptoModels service with optimized process termination.

        Args:
            force (bool): If True, force kill processes immediately without graceful termination.

        Returns:
            bool: True if the service stopped successfully, False otherwise.
        """
        if not os.path.exists(self.msgpack_file):
            logger.warning("No running CryptoModels service to stop.")
            return False

        try:
            # Load service details from the msgpack file
            with open(self.msgpack_file, "rb") as f:
                service_info = msgpack.load(f)
            
            hash_val = service_info.get("hash")
            pid = service_info.get("pid")
            app_pid = service_info.get("app_pid")
            app_port = service_info.get("app_port")
            local_ai_port = service_info.get("port")

            if force:
                logger.info(f"Force stopping CryptoModels service '{hash_val}' running on port {app_port} (AI PID: {pid}, API PID: {app_pid})...")
            else:
                logger.info(f"Stopping CryptoModels service '{hash_val}' running on port {app_port} (AI PID: {pid}, API PID: {app_pid})...")
            
            # Use the optimized termination methods with force parameter
            timeout = 0 if force else 15
            ai_stopped = self._terminate_process_safely(pid, "CryptoModels service", timeout=timeout, force=force)
            api_stopped = self._terminate_process_safely(app_pid, "API service", timeout=timeout, force=force)
            
            # Brief pause to allow system cleanup
            time.sleep(1)
            
            # Verify ports are freed
            ports_info = [(app_port, "API"), (local_ai_port, "AI")]
            ports_freed = self._check_ports_freed(ports_info, max_retries=3)
            
            # Clean up metadata file - ONLY rely on direct process verification, not return values
            metadata_cleaned = False
            
            # Direct verification: check if processes are actually dead
            ai_actually_dead = True
            api_actually_dead = True
            
            # Check AI process
            if pid:
                if psutil.pid_exists(pid):
                    try:
                        process = psutil.Process(pid)
                        status = process.status()
                        if status not in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD, psutil.STATUS_STOPPED]:
                            ai_actually_dead = False
                            logger.warning(f"AI process (PID: {pid}) still running with status: {status}")
                        else:
                            logger.info(f"AI process (PID: {pid}) confirmed dead with status: {status}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        # Process is gone or inaccessible, consider it dead
                        ai_actually_dead = True
                        logger.info(f"AI process (PID: {pid}) confirmed dead (no longer exists)")
                else:
                    logger.info(f"AI process (PID: {pid}) confirmed dead (PID doesn't exist)")
            
            # Check API process
            if app_pid:
                if psutil.pid_exists(app_pid):
                    try:
                        process = psutil.Process(app_pid)
                        status = process.status()
                        if status not in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD, psutil.STATUS_STOPPED]:
                            api_actually_dead = False
                            logger.warning(f"API process (PID: {app_pid}) still running with status: {status}")
                        else:
                            logger.info(f"API process (PID: {app_pid}) confirmed dead with status: {status}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        # Process is gone or inaccessible, consider it dead
                        api_actually_dead = True
                        logger.info(f"API process (PID: {app_pid}) confirmed dead (no longer exists)")
                else:
                    logger.info(f"API process (PID: {app_pid}) confirmed dead (PID doesn't exist)")
            
            # Only remove msgpack file if both processes are confirmed dead
            if ai_actually_dead and api_actually_dead:
                metadata_cleaned = self._cleanup_service_metadata(force=force)
                logger.info("All processes confirmed dead, removing service metadata file")
            else:
                logger.warning("Keeping service metadata file - not all processes confirmed dead")
                logger.warning(f"AI dead: {ai_actually_dead}, API dead: {api_actually_dead}")
                # Log the termination attempt results for debugging
                logger.info(f"Termination attempt results - AI stopped: {ai_stopped}, API stopped: {api_stopped}")
            
            # Determine overall success
            success = ai_stopped and api_stopped and metadata_cleaned
            
            if success:
                logger.info("CryptoModels service stopped successfully.")
                if not ports_freed:
                    logger.warning("Service stopped but some ports may still be in use temporarily")
            else:
                logger.error("CryptoModels service stop completed with some failures")
                logger.error(f"CryptoModels stopped: {ai_stopped}, API stopped: {api_stopped}, metadata cleaned: {metadata_cleaned}")
            
            return success

        except Exception as e:
            logger.error(f"Error stopping CryptoModels service: {str(e)}", exc_info=True)
            return False

    def _terminate_process_safely(self, pid: int, process_name: str, timeout: int = 15, use_process_group: bool = True, force: bool = False) -> bool:
        """
        Safely terminate a process with graceful fallback to force kill.
        
        Args:
            pid: Process ID to terminate
            process_name: Human-readable process name for logging
            timeout: Timeout for graceful termination in seconds
            use_process_group: Whether to try process group termination first
            force: If True, force kill processes immediately without graceful termination
            
        Returns:
            bool: True if process was terminated successfully, False otherwise
        """
        if not pid:
            logger.warning(f"No PID provided for {process_name}")
            return True
        
        # Quick existence check
        if not psutil.pid_exists(pid):
            logger.info(f"Process {process_name} (PID: {pid}) not found, assuming already stopped")
            return True
        
        try:
            process = psutil.Process(pid)
            
            # Check if already in terminal state
            try:
                status = process.status()
                if status in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD, psutil.STATUS_STOPPED]:
                    logger.info(f"Process {process_name} (PID: {pid}) already terminated (status: {status})")
                    # In force mode, also kill any remaining child processes
                    if force:
                        try:
                            children = process.children(recursive=True)
                            for child in children:
                                try:
                                    child.kill()
                                    logger.debug(f"Force killed child process {child.pid}")
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return True
            
            logger.info(f"Terminating {process_name} (PID: {pid})...")
            
            # Collect children before termination
            children = []
            try:
                children = process.children(recursive=True)
                logger.debug(f"Found {len(children)} child processes for {process_name}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            
            # Phase 1: Graceful termination (skip if force=True)
            if not force:
                try:
                    if use_process_group:
                        # Try process group termination first (more efficient)
                        try:
                            pgid = os.getpgid(pid)
                            os.killpg(pgid, signal.SIGTERM)
                            logger.debug(f"Sent SIGTERM to process group {pgid}")
                        except (ProcessLookupError, OSError, PermissionError):
                            # Fall back to individual process termination
                            process.terminate()
                            logger.debug(f"Sent SIGTERM to process {pid}")
                            
                            # Terminate children individually
                            for child in children:
                                try:
                                    child.terminate()
                                    logger.debug(f"Sent SIGTERM to child process {child.pid}")
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                    else:
                        # Individual process termination
                        process.terminate()
                        logger.debug(f"Sent SIGTERM to process {pid}")
                        
                        for child in children:
                            try:
                                child.terminate()
                                logger.debug(f"Sent SIGTERM to child process {child.pid}")
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    logger.info(f"Process {process_name} disappeared during termination")
                    return True
                
                # Wait for graceful termination with exponential backoff
                wait_time = 0.1
                elapsed = 0
                while elapsed < timeout and psutil.pid_exists(pid):
                    time.sleep(wait_time)
                    elapsed += wait_time
                    wait_time = min(wait_time * 1.5, 2.0)  # Cap at 2 seconds
                    
                    # Check if process became zombie
                    try:
                        if process.status() == psutil.STATUS_ZOMBIE:
                            logger.info(f"{process_name} became zombie, considering stopped")
                            return True
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        return True
            else:
                logger.info(f"Force mode enabled - skipping graceful termination for {process_name} (PID: {pid})")
            
            # Phase 2: Force termination if still running
            if psutil.pid_exists(pid):
                logger.warning(f"Force killing {process_name} (PID: {pid})")
                try:
                    # Refresh children list
                    children = []
                    try:
                        process = psutil.Process(pid)
                        children = process.children(recursive=True)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    
                    if use_process_group:
                        # Try process group kill
                        try:
                            pgid = os.getpgid(pid)
                            os.killpg(pgid, signal.SIGKILL)
                            logger.debug(f"Sent SIGKILL to process group {pgid}")
                        except (ProcessLookupError, OSError, PermissionError):
                            # Fall back to individual kill
                            process.kill()
                            logger.debug(f"Sent SIGKILL to process {pid}")
                            
                            for child in children:
                                try:
                                    child.kill()
                                    logger.debug(f"Sent SIGKILL to child process {child.pid}")
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                    else:
                        # Individual process kill
                        process.kill()
                        logger.debug(f"Sent SIGKILL to process {pid}")
                        
                        for child in children:
                            try:
                                child.kill()
                                logger.debug(f"Sent SIGKILL to child process {child.pid}")
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    logger.info(f"Process {process_name} disappeared during force kill")
                    return True
                
                # Final wait for force termination (shorter timeout)
                force_timeout = 5
                for _ in range(force_timeout * 10):  # 0.1s intervals
                    if not psutil.pid_exists(pid):
                        break
                    time.sleep(0.1)
            
            # Final status check
            success = not psutil.pid_exists(pid)
            if success:
                logger.info(f"{process_name} terminated successfully")
            else:
                try:
                    process = psutil.Process(pid)
                    status = process.status()
                    if status == psutil.STATUS_ZOMBIE:
                        logger.warning(f"{process_name} is zombie but considered stopped")
                        success = True
                    else:
                        logger.error(f"Failed to terminate {process_name} (PID: {pid}), status: {status}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    success = True
                    
            return success
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            logger.info(f"Process {process_name} (PID: {pid}) no longer accessible")
            return True
        except Exception as e:
            logger.error(f"Error terminating {process_name} (PID: {pid}): {e}")
            return False

    async def _terminate_process_safely_async(self, pid: int, process_name: str, timeout: int = 15, use_process_group: bool = True) -> bool:
        """
        Async version of _terminate_process_safely for use in async contexts.
        
        Args:
            pid: Process ID to terminate
            process_name: Human-readable process name for logging
            timeout: Timeout for graceful termination in seconds
            use_process_group: Whether to try process group termination first
            
        Returns:
            bool: True if process was terminated successfully, False otherwise
        """
        if not pid:
            logger.warning(f"No PID provided for {process_name}")
            return True
        
        # Quick existence check
        if not psutil.pid_exists(pid):
            logger.info(f"Process {process_name} (PID: {pid}) not found, assuming already stopped")
            return True
        
        try:
            process = psutil.Process(pid)
            
            # Check if already in terminal state
            try:
                status = process.status()
                if status in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD, psutil.STATUS_STOPPED]:
                    logger.info(f"Process {process_name} (PID: {pid}) already terminated (status: {status})")
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return True
            
            logger.info(f"Terminating {process_name} (PID: {pid})...")
            
            # Collect children before termination
            children = []
            try:
                children = process.children(recursive=True)
                logger.debug(f"Found {len(children)} child processes for {process_name}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            
            # Phase 1: Graceful termination
            try:
                if use_process_group:
                    # Try process group termination first
                    try:
                        pgid = os.getpgid(pid)
                        os.killpg(pgid, signal.SIGTERM)
                        logger.debug(f"Sent SIGTERM to process group {pgid}")
                    except (ProcessLookupError, OSError, PermissionError):
                        # Fall back to individual process termination
                        process.terminate()
                        logger.debug(f"Sent SIGTERM to process {pid}")
                        
                        for child in children:
                            try:
                                child.terminate()
                                logger.debug(f"Sent SIGTERM to child process {child.pid}")
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                else:
                    # Individual process termination
                    process.terminate()
                    logger.debug(f"Sent SIGTERM to process {pid}")
                    
                    for child in children:
                        try:
                            child.terminate()
                            logger.debug(f"Sent SIGTERM to child process {child.pid}")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                            
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                logger.info(f"Process {process_name} disappeared during termination")
                return True
            
            # Wait for graceful termination with async sleep
            wait_time = 0.1
            elapsed = 0
            while elapsed < timeout and psutil.pid_exists(pid):
                await asyncio.sleep(wait_time)
                elapsed += wait_time
                wait_time = min(wait_time * 1.5, 2.0)  # Cap at 2 seconds
                
                # Check if process became zombie
                try:
                    if process.status() == psutil.STATUS_ZOMBIE:
                        logger.info(f"{process_name} became zombie, considering stopped")
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    return True
            
            # Phase 2: Force termination if still running
            if psutil.pid_exists(pid):
                logger.warning(f"Force killing {process_name} (PID: {pid})")
                try:
                    # Refresh children list
                    children = []
                    try:
                        process = psutil.Process(pid)
                        children = process.children(recursive=True)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    
                    if use_process_group:
                        # Try process group kill
                        try:
                            pgid = os.getpgid(pid)
                            os.killpg(pgid, signal.SIGKILL)
                            logger.debug(f"Sent SIGKILL to process group {pgid}")
                        except (ProcessLookupError, OSError, PermissionError):
                            # Fall back to individual kill
                            process.kill()
                            logger.debug(f"Sent SIGKILL to process {pid}")
                            
                            for child in children:
                                try:
                                    child.kill()
                                    logger.debug(f"Sent SIGKILL to child process {child.pid}")
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                    else:
                        # Individual process kill
                        process.kill()
                        logger.debug(f"Sent SIGKILL to process {pid}")
                        
                        for child in children:
                            try:
                                child.kill()
                                logger.debug(f"Sent SIGKILL to child process {child.pid}")
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    logger.info(f"Process {process_name} disappeared during force kill")
                    return True
                
                # Final wait for force termination with async sleep
                force_timeout = 5
                for _ in range(force_timeout * 10):  # 0.1s intervals
                    if not psutil.pid_exists(pid):
                        break
                    await asyncio.sleep(0.1)
            
            # Final status check
            success = not psutil.pid_exists(pid)
            if success:
                logger.info(f"{process_name} terminated successfully")
            else:
                try:
                    process = psutil.Process(pid)
                    status = process.status()
                    if status == psutil.STATUS_ZOMBIE:
                        logger.warning(f"{process_name} is zombie but considered stopped")
                        success = True
                    else:
                        logger.error(f"Failed to terminate {process_name} (PID: {pid}), status: {status}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    success = True
                    
            return success
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            logger.info(f"Process {process_name} (PID: {pid}) no longer accessible")
            return True
        except Exception as e:
            logger.error(f"Error terminating {process_name} (PID: {pid}): {e}")
            return False

    def _cleanup_service_metadata(self, force: bool = False) -> bool:
        """
        Clean up service metadata file with proper error handling.
        
        Args:
            force: If True, remove file even if processes might still be running
            
        Returns:
            bool: True if cleanup was successful, False otherwise
        """
        try:
            if not os.path.exists(self.msgpack_file):
                logger.debug("Service metadata file already removed")
                return True
                
            if not force:
                # Verify that processes are actually stopped before cleanup
                try:
                    with open(self.msgpack_file, "rb") as f:
                        service_info = msgpack.load(f)
                    
                    pid = service_info.get("pid")
                    app_pid = service_info.get("app_pid")
                    
                    # Check if any processes are still running (excluding zombies)
                    running_processes = []
                    if pid and psutil.pid_exists(pid):
                        try:
                            process = psutil.Process(pid)
                            status = process.status()
                            if status not in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD, psutil.STATUS_STOPPED]:
                                running_processes.append(f"AI server (PID: {pid})")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    if app_pid and psutil.pid_exists(app_pid):
                        try:
                            process = psutil.Process(app_pid)
                            status = process.status()
                            if status not in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD, psutil.STATUS_STOPPED]:
                                running_processes.append(f"API server (PID: {app_pid})")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    
                    if running_processes:
                        logger.warning(f"Not cleaning up metadata - processes still running: {', '.join(running_processes)}")
                        return False
                        
                except Exception as e:
                    logger.warning(f"Could not verify process status, proceeding with cleanup: {e}")
            
            os.remove(self.msgpack_file)
            logger.info("Service metadata file removed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error removing service metadata file: {str(e)}")
            return False

    def _get_model_template_path(self, model_family: str) -> str:
        """Get the template path for a specific model family."""
        chat_template_path = pkg_resources.resource_filename("crypto_models", f"examples/templates/{model_family}.jinja")
        # check if the template file exists
        if not os.path.exists(chat_template_path):
            return None
        return chat_template_path

    def _get_model_best_practice_path(self, model_family: str) -> str:
        """Get the best practices for a specific model family."""
        best_practice_path = pkg_resources.resource_filename("crypto_models", f"examples/best_practices/{model_family}.json")
        # check if the best practices file exists
        if not os.path.exists(best_practice_path):
            return None
        return best_practice_path
    
    def _build_embed_command(self, model_path: str, port: int, host: str) -> list:
        """Build the embed command with common parameters."""
        command = [
            self.llama_server_path,
            "--model", str(model_path),
            "--port", str(port),
            "--host", host,
            "--embedding",
            "--pooling", "cls",
            "-ub", "8192",
            "-ngl", "9999"
        ]
        return command

    def _build_ai_command(self, model_path: str, port: int, host: str, context_length: int, template_path: Optional[str] = None, best_practice_path: Optional[str] = None) -> list:
        """Build the AI command with common parameters."""
        command = [
            self.llama_server_path,
            "--model", str(model_path),
            "--port", str(port),
            "--host", host,
            "-c", str(context_length),
            "-fa",
            "--pooling", "cls",
            "--embeddings",
            "--no-webui",
            "-ngl", "9999",
            "--jinja",
            "--reasoning-format", "none"
        ]
        
        if template_path:
            command.extend(["--chat-template-file", template_path])
        
        if best_practice_path:
            with open(best_practice_path, "r") as f:
                best_practice = json.load(f)
                for key, value in best_practice.items():
                    command.extend([f"--{key}", str(value)])
        return command

    
    def _check_multimodal_support(self, local_model_path: str) -> tuple[bool, Optional[str]]:
        """
        Check if model supports multimodal with single file check.
        Returns (is_multimodal, projector_path)
        """
        projector_path = f"{local_model_path}-projector"
        is_multimodal = os.path.exists(projector_path)
        return is_multimodal, projector_path if is_multimodal else None

    def _create_service_metadata(self, hash: str, local_model_path: str, local_ai_port: int, 
                                port: int, context_length: int, task: str, 
                                is_multimodal: bool, projector_path: Optional[str]) -> dict:
        """Create service metadata dictionary with all required fields."""
        return {
            "task": task,
            "hash": hash,
            "port": local_ai_port,
            "local_text_path": local_model_path,
            "app_port": port,
            "context_length": context_length,
            "last_activity": time.time(),
            "multimodal": is_multimodal,
            "local_projector_path": projector_path
        }

    def _validate_lock_data(self, lock_data: dict) -> tuple[bool, str]:
        """
        Validate lock file data structure and return (is_valid, error_message).
        
        Args:
            lock_data: Dictionary containing lock file data
            
        Returns:
            tuple: (is_valid: bool, error_message: str)
        """
        required_fields = ["pid", "timestamp", "hash", "host", "port"]
        
        for field in required_fields:
            if field not in lock_data:
                return False, f"Missing required field: {field}"
            
        # Validate field types
        if not isinstance(lock_data["pid"], int) or lock_data["pid"] <= 0:
            return False, "Invalid PID: must be positive integer"
            
        if not isinstance(lock_data["timestamp"], (int, float)) or lock_data["timestamp"] <= 0:
            return False, "Invalid timestamp: must be positive number"
            
        if not isinstance(lock_data["hash"], str) or not lock_data["hash"].strip():
            return False, "Invalid hash: must be non-empty string"
            
        if not isinstance(lock_data["host"], str) or not lock_data["host"].strip():
            return False, "Invalid host: must be non-empty string"
            
        if not isinstance(lock_data["port"], int) or not (1 <= lock_data["port"] <= 65535):
            return False, "Invalid port: must be integer between 1-65535"
            
        # Check version compatibility (if present)
        version = lock_data.get("version", "1.0")
        if not isinstance(version, str):
            return False, "Invalid version: must be string"
            
        # For now, we support version 1.0 (current) and future versions
        try:
            major, minor = map(int, version.split(".")[:2])
            if major > 1:
                return False, f"Unsupported lock file version: {version}"
        except (ValueError, IndexError):
            return False, f"Invalid version format: {version}"
            
        return True, ""
    
    def _acquire_start_lock(self, hash: str, host: str, port: int) -> bool:
        """
        Acquire the start lock efficiently with atomic operations and minimal race conditions.
        Returns True if lock acquired, False otherwise.
        """
        current_pid = os.getpid()
        current_time = time.time()
        
        # Create lock data
        lock_data = {
            "version": "1.0",
            "pid": current_pid,
            "hash": hash,
            "timestamp": current_time,
            "host": host,
            "port": port
        }
        
        # Attempt to acquire lock with atomic operations
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Check and handle existing lock
                if self.start_lock_file.exists():
                    try:
                        with open(self.start_lock_file, "r") as f:
                            existing_lock_data = json.load(f)
                        
                        # Validate existing lock data
                        is_valid, error_msg = self._validate_lock_data(existing_lock_data)
                        if not is_valid:
                            logger.warning(f"Removing corrupted lock file (attempt {attempt + 1}/{max_attempts}): {error_msg}")
                            try:
                                self.start_lock_file.unlink()
                            except OSError:
                                pass  # May have been removed by another process
                        elif existing_lock_data.get("pid") == current_pid:
                            # Same process already has the lock
                            logger.info(f"Lock already acquired by current process (PID: {current_pid})")
                            return True
                        elif existing_lock_data.get("pid") and not psutil.pid_exists(existing_lock_data["pid"]):
                            logger.warning(f"Removing lock file for dead process (PID: {existing_lock_data['pid']})")
                            try:
                                self.start_lock_file.unlink()
                            except OSError:
                                pass  # May have been removed by another process
                        else:
                            # Check if it's actually the same process (edge case)
                            if existing_lock_data.get("pid") and psutil.pid_exists(existing_lock_data["pid"]):
                                logger.error(f"Another process (PID: {existing_lock_data['pid']}) is already starting service")
                                return False
                            else:
                                # Process doesn't exist, remove stale lock
                                logger.warning(f"Removing lock file for non-existent process (PID: {existing_lock_data['pid']})")
                                try:
                                    self.start_lock_file.unlink()
                                except OSError:
                                    pass
                                    
                    except (json.JSONDecodeError, KeyError, OSError) as e:
                        logger.warning(f"Corrupted lock file (attempt {attempt + 1}/{max_attempts}), removing: {e}")
                        try:
                            self.start_lock_file.unlink()
                        except OSError:
                            pass
                
                # Atomic lock creation using temporary file + rename
                temp_lock_file = self.start_lock_file.with_suffix('.tmp')
                try:
                    # Write to temporary file first
                    with open(temp_lock_file, "w") as f:
                        json.dump(lock_data, f)
                        f.flush()
                        os.fsync(f.fileno())  # Ensure data is written to disk
                    
                    # Atomic rename (on most filesystems)
                    temp_lock_file.rename(self.start_lock_file)
                    logger.info(f"Acquired start lock for PID {current_pid}")
                    return True
                    
                except OSError as e:
                    # Clean up temp file on error
                    try:
                        temp_lock_file.unlink()
                    except OSError:
                        pass
                    
                    if attempt < max_attempts - 1:
                        logger.warning(f"Failed to create lock file (attempt {attempt + 1}/{max_attempts}): {e}")
                        time.sleep(0.1 * (attempt + 1))  # Brief exponential backoff
                        continue
                    else:
                        logger.error(f"Failed to create lock file after {max_attempts} attempts: {e}")
                        return False
                        
            except Exception as e:
                logger.error(f"Unexpected error during lock acquisition (attempt {attempt + 1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                else:
                    return False
        
        return False
    
    def _release_start_lock(self) -> None:
        """Release the start lock if we own it, with improved error handling."""
        if not self.start_lock_file.exists():
            logger.debug("Lock file does not exist, nothing to release")
            return
            
        current_pid = os.getpid()
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                # Read and verify lock ownership
                with open(self.start_lock_file, "r") as f:
                    lock_data = json.load(f)
                
                # Validate lock data
                is_valid, error_msg = self._validate_lock_data(lock_data)
                if not is_valid:
                    logger.warning(f"Corrupted lock file during release (attempt {attempt + 1}/{max_attempts}): {error_msg}")
                    try:
                        self.start_lock_file.unlink()
                        logger.info("Removed corrupted lock file during release")
                        return
                    except OSError as e2:
                        if attempt < max_attempts - 1:
                            logger.warning(f"Failed to remove corrupted lock file: {e2}")
                            time.sleep(0.1 * (attempt + 1))
                            continue
                        else:
                            logger.error(f"Failed to remove corrupted lock file after {max_attempts} attempts: {e2}")
                            return
                
                existing_pid = lock_data.get("pid")
                if not existing_pid:
                    logger.warning("Lock file has no PID, removing corrupted file")
                    try:
                        self.start_lock_file.unlink()
                        logger.info("Removed corrupted lock file")
                    except OSError as e:
                        logger.warning(f"Failed to remove corrupted lock file: {e}")
                    return
                
                if existing_pid == current_pid:
                    # We own the lock, remove it
                    try:
                        self.start_lock_file.unlink()
                        logger.info(f"Released start lock for PID {current_pid}")
                    except OSError as e:
                        if attempt < max_attempts - 1:
                            logger.warning(f"Failed to remove lock file (attempt {attempt + 1}/{max_attempts}): {e}")
                            time.sleep(0.1 * (attempt + 1))
                            continue
                        else:
                            logger.error(f"Failed to remove lock file after {max_attempts} attempts: {e}")
                    return
                else:
                    # Check if the other process still exists
                    if psutil.pid_exists(existing_pid):
                        logger.warning(f"Lock file owned by different process (PID: {existing_pid}), not removing")
                    else:
                        logger.warning(f"Lock file owned by dead process (PID: {existing_pid}), removing stale lock")
                        try:
                            self.start_lock_file.unlink()
                            logger.info(f"Removed stale lock file for dead process (PID: {existing_pid})")
                        except OSError as e:
                            logger.warning(f"Failed to remove stale lock file: {e}")
                    return
                    
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Corrupted lock file during release (attempt {attempt + 1}/{max_attempts}): {e}")
                try:
                    self.start_lock_file.unlink()
                    logger.info("Removed corrupted lock file during release")
                    return
                except OSError as e2:
                    if attempt < max_attempts - 1:
                        logger.warning(f"Failed to remove corrupted lock file: {e2}")
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    else:
                        logger.error(f"Failed to remove corrupted lock file after {max_attempts} attempts: {e2}")
                        return
                        
            except OSError as e:
                if attempt < max_attempts - 1:
                    logger.warning(f"Error accessing lock file during release (attempt {attempt + 1}/{max_attempts}): {e}")
                    time.sleep(0.1 * (attempt + 1))
                    continue
                else:
                    logger.error(f"Error accessing lock file during release after {max_attempts} attempts: {e}")
                    return
            
            except Exception as e:
                logger.error(f"Unexpected error during lock release (attempt {attempt + 1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                else:
                    return

    def _check_port_availability(self, host: str, port: int, timeout: int = 2) -> bool:
        """
        Check if a port is available (not in use).
        Returns True if port is free, False if in use.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                result = s.connect_ex((host, port))
                return result != 0  # 0 means connection successful (port in use)
        except socket.error:
            return True  # Assume port is free if we can't check
        
    def _check_ports_freed(self, ports_info: list, max_retries: int = 5) -> bool:
        """
        Check if multiple ports are freed with optimized retry logic.
        ports_info: list of (port, service_name) tuples
        Returns True if all ports are freed.
        """
        if not ports_info:
            return True
            
        freed_ports = set()
        
        for retry in range(max_retries):
            for port, service_name in ports_info:
                if port and port not in freed_ports:
                    if self._check_port_availability('localhost', port, timeout=1):
                        freed_ports.add(port)
                        logger.debug(f"{service_name} port {port} is now free")
            
            # If all ports are freed, return early
            if len(freed_ports) == len([p for p, _ in ports_info if p]):
                return True
                
            if retry < max_retries - 1:
                time.sleep(1)
        
        # Log which ports are still in use
        for port, service_name in ports_info:
            if port and port not in freed_ports:
                logger.warning(f"{service_name} port {port} still in use after {max_retries} checks")
        
        return len(freed_ports) == len([p for p, _ in ports_info if p])
    
    async def kill_ai_server(self) -> bool:
        """Kill the AI server process if it's running (optimized async version)."""
        try:
            if not os.path.exists(self.msgpack_file):
                logger.warning("No service info found, cannot kill AI server")
                return False
                
            # Load service details from the msgpack file
            with open(self.msgpack_file, "rb") as f:
                service_info = msgpack.load(f)
                
            pid = service_info.get("pid")
            if not pid:
                logger.warning("No PID found in service info, cannot kill AI server")
                return False
                
            logger.info(f"Attempting to kill AI server with PID {pid}")
            
            # Use the optimized async termination method
            success = await self._terminate_process_safely_async(pid, "AI server", timeout=15)
            
            # Clean up service info if process was successfully killed
            if success:
                try:
                    # Remove PID from service info to indicate server is no longer running
                    service_info.pop("pid", None)
                    
                    with open(self.msgpack_file, "wb") as f:
                        msgpack.dump(service_info, f)
                    
                    logger.info("AI server stopped successfully and service info cleaned up")
                except Exception as e:
                    logger.warning(f"AI server stopped but failed to clean up service info: {str(e)}")
                
                return True
            else:
                logger.error("Failed to stop AI server")
                return False
            
        except Exception as e:
            logger.error(f"Error killing AI server: {str(e)}", exc_info=True)
            return False
    
    async def reload_ai_server(self, service_start_timeout: int = 120) -> bool:
        """Reload the AI server process (async version for API usage)."""
        try:
            if not os.path.exists(self.msgpack_file):
                logger.error("No service info found, cannot reload AI server")
                return False
                
            # Load service details from the msgpack file
            with open(self.msgpack_file, "rb") as f:
                service_info = msgpack.load(f)
                
            running_ai_command = service_info.get("running_ai_command")
            if not running_ai_command:
                logger.error("No running_ai_command found in service info, cannot reload AI server")
                return False
                
            logger.info(f"Reloading AI server with command: {running_ai_command}")

            ai_log_stderr = self.logs_dir / "ai.log"
            
            try:
                with open(ai_log_stderr, 'w') as stderr_log:
                    ai_process = subprocess.Popen(
                        running_ai_command,
                        stderr=stderr_log,
                        preexec_fn=os.setsid
                    )
                logger.info(f"AI logs written to {ai_log_stderr}")
            except Exception as e:
                logger.error(f"Error starting CryptoModels service: {str(e)}", exc_info=True)
                return False
            
            # Wait for the process to start by checking the health endpoint
            port = service_info["port"]
            if not wait_for_health(port, timeout=service_start_timeout):
                logger.error(f"AI server failed to start within {service_start_timeout} seconds")
                return False
            
            # Check if the process is running
            if ai_process.poll() is None:
                # Update the service info with new PID
                service_info["pid"] = ai_process.pid
                with open(self.msgpack_file, "wb") as f:
                    msgpack.dump(service_info, f)
                logger.info(f"Successfully reloaded AI server with PID {ai_process.pid}")
                return True
            else:
                logger.error(f"Failed to reload AI server: Process exited with code {ai_process.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"Error reloading AI server: {str(e)}", exc_info=True)
            return False
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service info from msgpack file with error handling."""
        if not os.path.exists(self.msgpack_file):
            raise CryptoAgentsServiceError("Service information not available")
        
        try:
            with open(self.msgpack_file, "rb") as f:
                return msgpack.load(f)
        except Exception as e:
            raise CryptoAgentsServiceError(f"Failed to load service info: {str(e)}")
    
    def update_service_info(self, updates: Dict[str, Any]) -> bool:
        """Update service information in the msgpack file."""
        try:
            if os.path.exists(self.msgpack_file):
                with open(self.msgpack_file, "rb") as f:
                    service_info = msgpack.load(f)
            else:
                service_info = {}
            
            service_info.update(updates)
            
            with open(self.msgpack_file, "wb") as f:
                msgpack.dump(service_info, f)
            
            return True
        except Exception as e:
            logger.error(f"Failed to update service info: {str(e)}")
            return False

    def _build_image_generation_command(self, model_path: str, port: int, host: str, config_name: str, 
                                       lora_paths: Optional[List[str]] = None, 
                                       lora_scales: Optional[List[float]] = None) -> list:
        """Build the image-generation command with MLX Flux parameters and optional LoRA support."""
        command = [
            "mlx-flux",
            "serve",
            "--model-path", str(model_path),
            "--config-name", config_name,
            "--port", str(port),
            "--host", host
        ]
        
        # Validate LoRA parameters
        if lora_paths and lora_scales and len(lora_paths) != len(lora_scales):
            raise ValueError(f"LoRA paths count ({len(lora_paths)}) must match scales count ({len(lora_scales)})")
        
        # Add LoRA paths if provided
        if lora_paths:
            lora_path_str = ",".join(lora_paths)
            command.extend(["--lora-paths", lora_path_str])
        
        # Add LoRA scales if provided
        if lora_scales:
            lora_scale_str = ",".join(str(scale) for scale in lora_scales)
            command.extend(["--lora-scales", lora_scale_str])
        
        return command

    def _build_model_command(self, folder_name: str, local_model_path: str, local_ai_port: int, host: str, context_length: int) -> list:
        """Build the appropriate command for a model based on its family."""
        if "gemma-3n" in folder_name.lower():
            template_path, best_practice_path = self._get_family_template_and_practice("gemma-3n")
            return self._build_ai_command(
                local_model_path, local_ai_port, host, context_length, template_path
            )
        elif "gemma-3" in folder_name.lower():
            context_length = context_length // 2
            template_path, best_practice_path = self._get_family_template_and_practice("gemma-3")
            return self._build_ai_command(
                local_model_path, local_ai_port, host, context_length, template_path
            )
        elif "lfm2" in folder_name.lower():
            template_path, best_practice_path = self._get_family_template_and_practice("lfm2")
            return self._build_ai_command(
                local_model_path, local_ai_port, host, context_length, template_path, best_practice_path
            )
        elif "devstral-small" in folder_name.lower():
            template_path, best_practice_path = self._get_family_template_and_practice("devstral-small")
            return self._build_ai_command(
                local_model_path, local_ai_port, host, context_length, template_path, best_practice_path
            )
        elif "openreasoning-nemotron" in folder_name.lower():
            template_path, best_practice_path = self._get_family_template_and_practice("openreasoning-nemotron")
            return self._build_ai_command(
                local_model_path, local_ai_port, host, context_length, template_path, best_practice_path
            )
        elif "qwen25" in folder_name.lower():
            template_path, best_practice_path = self._get_family_template_and_practice("qwen25")
            return self._build_ai_command(
                local_model_path, local_ai_port, host, context_length, template_path, best_practice_path
            )
        elif "qwen3" in folder_name.lower():
            template_path, best_practice_path = self._get_family_template_and_practice("qwen3")
            return self._build_ai_command(
                local_model_path, local_ai_port, host, context_length, template_path, best_practice_path
            )
        elif "llama" in folder_name.lower():
            template_path, best_practice_path = self._get_family_template_and_practice("llama")
            return self._build_ai_command(
                local_model_path, local_ai_port, host, context_length, template_path, best_practice_path
            )
        else:
            return self._build_ai_command(
                local_model_path, local_ai_port, host, context_length
            )

    async def switch_model(self, target_hash: str, service_start_timeout: int = 120) -> bool:
        """
        Switch to a different model that was registered during multi-model start.
        This will offload the currently active model and load the requested model.

        Args:
            target_hash (str): Hash of the model to switch to.
            service_start_timeout (int): Timeout for service startup in seconds.

        Returns:
            bool: True if model switch was successful, False otherwise.
        """
        try:
            if not os.path.exists(self.msgpack_file):
                logger.error("No service info found, cannot switch model")
                return False

            # Load service details
            with open(self.msgpack_file, "rb") as f:
                service_info = msgpack.load(f)

            # Check if target model is available
            models = service_info.get("models", {})
            if target_hash not in models:
                logger.error(f"Model {target_hash} not found in available models")
                return False

            # Check if model is already active
            if models[target_hash].get("active", False):
                logger.info(f"Model {target_hash} is already active")
                return True

            target_model = models[target_hash]
            logger.info(f"Switching to model: {target_hash}")

            # Kill current AI server process
            if not await self.kill_ai_server():
                logger.error("Failed to stop current AI server")
                return False

            # Build command for target model
            local_model_path = target_model["local_model_path"]
            metadata = target_model["metadata"]
            folder_name = metadata.get("folder_name", "")
            task = metadata.get("task", "chat")
            config_name = metadata.get("config_name", "flux-dev")
            is_lora = metadata.get("lora", False)
            
            # Get current service configuration
            local_ai_port = service_info["port"]
            host = service_info.get("host", "localhost")
            context_length = service_info.get("context_length", 32768)

            # Build appropriate command based on task
            if task == "embed":
                running_ai_command = self._build_embed_command(local_model_path, local_ai_port, host)
            elif task == "image-generation":
                if not shutil.which("mlx-flux"):
                    raise CryptoAgentsServiceError("mlx-flux command not found in PATH")
                
                # Initialize LoRA variables
                lora_paths = None
                lora_scales = None
                effective_model_path = local_model_path
                
                if is_lora:
                    base_model_path = target_model["base_model_path"]
                    lora_config = target_model["lora_config"]       
                    effective_model_path = base_model_path
                    lora_paths = []
                    lora_scales = []
                    for i in range(len(lora_config)):
                        lora_paths.append(lora_config[str(i)]["path"])
                        lora_scales.append(lora_config[str(i)]["scale"])

                    
                
                running_ai_command = self._build_image_generation_command(
                    effective_model_path, local_ai_port, host, config_name, lora_paths, lora_scales
                )
            else:
                running_ai_command = self._build_model_command(folder_name, local_model_path, local_ai_port, host, context_length)
                
                # Add multimodal support if available
                is_multimodal, projector_path = self._check_multimodal_support(local_model_path)
                if is_multimodal:
                    running_ai_command.extend([
                        "--mmproj", str(projector_path)
                    ])

            logger.info(f"Starting new model with command: {running_ai_command}")

            # Start new AI server process
            ai_log_stderr = self.logs_dir / "ai.log"
            try:
                with open(ai_log_stderr, 'w') as stderr_log:
                    ai_process = subprocess.Popen(
                        running_ai_command,
                        stderr=stderr_log,
                        preexec_fn=os.setsid
                    )
                logger.info(f"AI logs written to {ai_log_stderr}")
            except Exception as e:
                logger.error(f"Error starting new model: {str(e)}", exc_info=True)
                return False

            # Wait for the new process to start
            if not wait_for_health(local_ai_port, timeout=service_start_timeout):
                logger.error(f"New model failed to start within {service_start_timeout} seconds")
                return False

            # Update service metadata
            service_info["hash"] = target_hash
            service_info["pid"] = ai_process.pid
            service_info["running_ai_command"] = running_ai_command
            service_info["local_text_path"] = local_model_path
            service_info["family"] = metadata.get("family", None)
            service_info["folder_name"] = folder_name
            service_info["ram"] = metadata.get("ram", None)
            service_info["task"] = task
            service_info["multimodal"] = target_model.get("multimodal", False)
            service_info["local_projector_path"] = target_model.get("local_projector_path")

            # Update active model flags
            for hash_val in models:
                models[hash_val]["active"] = (hash_val == target_hash)

            # Save updated service info
            with open(self.msgpack_file, "wb") as f:
                msgpack.dump(service_info, f)

            logger.info(f"Successfully switched to model {target_hash} with PID {ai_process.pid}")
            return True

        except Exception as e:
            logger.error(f"Error switching model: {str(e)}", exc_info=True)
            return False

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available models in the current service.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping model hashes to their information.
        """
        try:
            service_info = self.get_service_info()
            return service_info.get("models", {})
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return {}

    def get_active_model(self) -> Optional[str]:
        """
        Get the hash of the currently active model.

        Returns:
            Optional[str]: Hash of the active model, or None if no model is active.
        """
        try:
            models = self.get_available_models()
            for hash_val, model_info in models.items():
                if model_info.get("active", False):
                    return hash_val
            return None
        except Exception as e:
            logger.error(f"Error getting active model: {str(e)}")
            return None

    def _validate_lora_metadata(self, lora_metadata: dict) -> tuple[bool, str]:
        """Validate LoRA metadata structure and return validation result."""
        required_fields = ["base_model", "lora_paths", "lora_scales"]
        
        for field in required_fields:
            if field not in lora_metadata:
                return False, f"Missing required field: {field}"
        
        lora_paths = lora_metadata["lora_paths"]
        lora_scales = lora_metadata["lora_scales"]
        
        if not isinstance(lora_paths, list):
            return False, "lora_paths must be a list"
        
        if not isinstance(lora_scales, list):
            return False, "lora_scales must be a list"
        
        if len(lora_paths) != len(lora_scales):
            return False, f"lora_paths count ({len(lora_paths)}) must match lora_scales count ({len(lora_scales)})"
        
        if len(lora_paths) == 0:
            return False, "lora_paths cannot be empty"
        
        # Validate that all scales are numeric
        for i, scale in enumerate(lora_scales):
            if not isinstance(scale, (int, float)):
                return False, f"lora_scales[{i}] must be a number, got {type(scale).__name__}"
        
        return True, "Valid LoRA metadata"
    
    def _load_lora_metadata(self, metadata_path: str) -> tuple[Optional[dict], Optional[str]]:
        """Load and validate LoRA metadata from file."""
        try:
            with open(metadata_path, "r") as f:
                lora_metadata = json.load(f)
            
            is_valid, message = self._validate_lora_metadata(lora_metadata)
            if not is_valid:
                logger.error(f"Invalid LoRA metadata: {message}")
                return None, f"Invalid LoRA metadata: {message}"
            
            return lora_metadata, None
        
        except FileNotFoundError:
            error_msg = f"LoRA metadata file not found: {metadata_path}"
            logger.error(error_msg)
            return None, error_msg
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in LoRA metadata file: {e}"
            logger.error(error_msg)
            return None, error_msg
        except Exception as e:
            error_msg = f"Error loading LoRA metadata: {e}"
            logger.error(error_msg)
            return None, error_msg