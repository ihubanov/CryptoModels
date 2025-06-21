import os
import json
import time
import signal
import pickle
import psutil
import asyncio
import socket
import requests
import subprocess
import pkg_resources
from pathlib import Path
from loguru import logger
from typing import Optional, Dict, Any
from local_ai.utils import wait_for_health
from local_ai.download import download_model_from_filecoin_async

class LocalAIServiceError(Exception):
    """Base exception for Local AI service errors."""
    pass

class ServiceStartError(LocalAIServiceError):
    """Exception raised when service fails to start."""
    pass

class ModelNotFoundError(LocalAIServiceError):
    """Exception raised when model file is not found."""
    pass

class LocalAIManager:
    """Manages a local AI service with optimized performance."""
    
    # Performance constants
    LOCK_TIMEOUT = 1800  # 30 minutes
    PORT_CHECK_TIMEOUT = 2
    HEALTH_CHECK_TIMEOUT = 300  # 5 minutes  
    PROCESS_TERM_TIMEOUT = 15
    MAX_PORT_RETRIES = 5
    
    def __init__(self):
        """Initialize the LocalAIManager with optimized defaults."""       
        self.pickle_file = Path(os.getenv("RUNNING_SERVICE_FILE", "running_service.pkl"))
        self.loaded_models: Dict[str, Any] = {}
        self.llama_server_path = os.getenv("LLAMA_SERVER")
        if not self.llama_server_path or not os.path.exists(self.llama_server_path):
            raise LocalAIServiceError("llama-server executable not found in LLAMA_SERVER or PATH")
        self.start_lock_file = Path(os.getenv("START_LOCK_FILE", "start_lock.lock"))
        
    def _get_free_port(self) -> int:
        """Get a free port number."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    
    def restart(self):
        """
        Restart the currently running AI service.

        Returns:
            bool: True if the service restarted successfully, False otherwise.
        """
        if not self.pickle_file.exists():
            logger.warning("No running AI service to restart.")
            return False
        
        try:
            # Load service details from the pickle file
            with open(self.pickle_file, "rb") as f:
                service_info = pickle.load(f)
            
            hash = service_info.get("hash")
            port = service_info.get("app_port")
            context_length = service_info.get("context_length")

            logger.info(f"Restarting AI service '{hash}' running on port {port}...")

            # Stop the current service
            self.stop()

            # Start the service with the same parameters
            return self.start(hash, port, context_length=context_length)
        except Exception as e:
            logger.error(f"Error restarting AI service: {str(e)}", exc_info=True)
            return False

    def _get_family_template_and_practice(self, model_family: str):
        """Helper to get template and best practice paths based on folder name."""
        return (
            self._get_model_template_path(model_family),
            self._get_model_best_practice_path(model_family)
        )
    
    def _retry_request_json(self, url: str, retries: int = 2, delay: int = 2, timeout: int = 8) -> Optional[dict]:
        """
        Utility to retry a GET request for JSON data with optimized parameters.
        Returns parsed JSON data or None on failure.
        """
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

    def start(self, hash: str, port: int = 11434, host: str = "0.0.0.0", context_length: int = 32768) -> bool:
        """
        Start the local AI service in the background.

        Args:
            hash (str): Filecoin hash of the model to download and run.
            port (int): Port number for the AI service (default: 11434).
            host (str): Host address for the AI service (default: "0.0.0.0").
            context_length (int): Context length for the model (default: 32768).

        Returns:
            bool: True if service started successfully, False otherwise.

        Raises:
            ValueError: If hash is not provided when no model is running.
            ModelNotFoundError: If model file is not found.
            ServiceStartError: If service fails to start.
        """
        if not hash:
            raise ValueError("Filecoin hash is required to start the service")

        # Acquire process lock to prevent concurrent starts
        if not self._acquire_start_lock(hash, host, port):
            return False
        
        try:
            # Check if the requested port is available before doing expensive operations
            if not self._check_port_availability(host, port):
                raise ServiceStartError(f"Port {port} is already in use on {host}")
                # Connection refused means port is free, which is what we want

            try:
                logger.info(f"Starting local AI service for model with hash: {hash}")
                
                local_model_path = asyncio.run(download_model_from_filecoin_async(hash))
                local_projector_path = local_model_path + "-projector"
                model_running = self.get_running_model()
                local_ai_port = self._get_free_port()
                if model_running:
                    if model_running == hash:
                        logger.warning(f"Model '{hash}' already running on port {port}")
                        return True
                    logger.info(f"Stopping existing model '{model_running}' on port {port}")
                    self.stop()

                if not os.path.exists(local_model_path):
                    raise ModelNotFoundError(f"Model file not found at: {local_model_path}")

                # Optimized metadata and multimodal checking
                model_dir = os.path.dirname(local_model_path)
                folder_name, metadata = self._load_or_fetch_metadata(hash, model_dir)
                is_multimodal, projector_path = self._check_multimodal_support(local_model_path)
                
                service_metadata = self._create_service_metadata(
                    hash, local_model_path, local_ai_port, port, context_length, 
                    metadata, is_multimodal, projector_path
                )

                if "gemma" in folder_name.lower():
                    template_path, best_practice_path = self._get_family_template_and_practice("gemma")
                    # Gemma models are memory intensive, so we reduce the context length
                    context_length = context_length // 2
                    running_ai_command = self._build_ai_command(
                        local_model_path, local_ai_port, host, context_length, template_path
                    )
                elif "qwen25" in folder_name.lower():
                    template_path, best_practice_path = self._get_family_template_and_practice("qwen25")
                    running_ai_command = self._build_ai_command(
                        local_model_path, local_ai_port, host, context_length, template_path, best_practice_path
                    )
                elif "qwen3" in folder_name.lower():
                    template_path, best_practice_path = self._get_family_template_and_practice("qwen3")
                    running_ai_command = self._build_ai_command(
                        local_model_path, local_ai_port, host, context_length, template_path, best_practice_path
                    )
                elif "llama" in folder_name.lower():
                    template_path, best_practice_path = self._get_family_template_and_practice("llama")
                    running_ai_command = self._build_ai_command(
                        local_model_path, local_ai_port, host, context_length, template_path, best_practice_path
                    )
                else:
                    running_ai_command = self._build_ai_command(
                        local_model_path, local_ai_port, host, context_length
                    )

                if service_metadata["multimodal"]:
                    running_ai_command.extend([
                        "--mmproj", str(local_projector_path)
                    ])

                logger.info(f"Starting process: {' '.join(running_ai_command)}")
                service_metadata["running_ai_command"] = running_ai_command
                # Create log files for stdout and stderr for AI process
                os.makedirs("logs", exist_ok=True)
                ai_log_stderr = Path(f"logs/ai.log")
                ai_process = None
                try:
                    with open(ai_log_stderr, 'w') as stderr_log:
                        ai_process = subprocess.Popen(
                            running_ai_command,
                            stderr=stderr_log,
                            preexec_fn=os.setsid
                        )
                    logger.info(f"AI logs written to {ai_log_stderr}")
                except Exception as e:
                    logger.error(f"Error starting AI service: {str(e)}", exc_info=True)
                    return False
        
                if not wait_for_health(local_ai_port):
                    logger.error(f"Service failed to start within 600 seconds")
                    ai_process.terminate()
                    return False
                
                logger.info(f"[LOCAL-AI] Local AI service started on port {local_ai_port}")

                # start the FastAPI app in the background           
                uvicorn_command = [
                    "uvicorn",
                    "local_ai.apis:app",
                    "--host", host,
                    "--port", str(port),
                    "--log-level", "info"
                ]
                logger.info(f"Starting process: {' '.join(uvicorn_command)}")
                # Create log files for stdout and stderr
                os.makedirs("logs", exist_ok=True)
                api_log_stderr = Path(f"logs/api.log")
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
                    ai_process.terminate()
                    return False
                
                if not wait_for_health(port):
                    logger.error(f"API service failed to start within 600 seconds")
                    ai_process.terminate()
                    apis_process.terminate()
                    return False

                logger.info(f"Service started on port {port} for model: {hash}")

                # Update service metadata with process IDs
                service_metadata.update({
                    "pid": ai_process.pid,
                    "app_pid": apis_process.pid
                })

                self._dump_running_service(service_metadata)    

                # update service metadata to the FastAPI app
                try:
                    update_url = f"http://localhost:{port}/update"
                    response = requests.post(update_url, json=service_metadata, timeout=10)
                    response.raise_for_status()  # Raise exception for HTTP error responses
                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to update service metadata: {str(e)}")
                    # Stop the partially started service
                    self.stop()
                    return False
                
                return True

            except Exception as e:
                logger.error(f"Error starting AI service: {str(e)}", exc_info=True)
                return False
                
        finally:
            # Always remove the lock when done (success or failure)
            self._release_start_lock()

    def _dump_running_service(self, metadata: dict):
        """Dump the running service details to a file."""
        try:
            with open(self.pickle_file, "wb") as f:
                pickle.dump(metadata, f)
        except Exception as e:
            logger.error(f"Error dumping running service: {str(e)}", exc_info=True)
            return False

    def get_running_model(self) -> Optional[str]:
        """
        Get currently running model hash if the service is healthy.

        Returns:
            Optional[str]: Running model hash or None if no healthy service exists.
        """
        if not self.pickle_file.exists():
            return None

        try:
            # Load service info from pickle file
            with open(self.pickle_file, "rb") as f:
                service_info = pickle.load(f)   
            model_hash = service_info.get("hash")      
            return model_hash

        except Exception as e:
            logger.error(f"Error getting running model: {str(e)}")
            return None
    
    def stop(self) -> bool:
        """
        Stop the running AI service.

        Returns:
            bool: True if the service stopped successfully, False otherwise.
        """
        if not os.path.exists(self.pickle_file):
            logger.warning("No running AI service to stop.")
            return False

        try:
            # Load service details from the pickle file
            with open(self.pickle_file, "rb") as f:
                service_info = pickle.load(f)
            
            hash_val = service_info.get("hash")
            pid = service_info.get("pid")
            app_pid = service_info.get("app_pid")
            app_port = service_info.get("app_port")
            local_ai_port = service_info.get("port")

            logger.info(f"Stopping AI service '{hash_val}' running on port {app_port} (AI PID: {pid}, API PID: {app_pid})...")
            
            def terminate_process_group(pid, process_name, timeout=15):
                """
                Terminate a process and its entire process group efficiently.
                Returns True if successful, False otherwise.
                """
                if not pid:
                    logger.warning(f"No PID provided for {process_name}")
                    return True
                
                # Quick check if process exists
                if not psutil.pid_exists(pid):
                    logger.info(f"Process {process_name} (PID: {pid}) not found, assuming already stopped")
                    return True
                
                try:
                    process = psutil.Process(pid)
                    
                    # Check if already terminated
                    status = process.status()
                    if status in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD, psutil.STATUS_STOPPED]:
                        logger.info(f"Process {process_name} (PID: {pid}) already terminated (status: {status})")
                        return True
                    
                    logger.info(f"Terminating {process_name} (PID: {pid})...")
                    
                    # Get children before termination
                    children = process.children(recursive=True)
                    
                    # Try graceful termination first
                    try:
                        # Terminate the main process
                        process.terminate()
                        logger.debug(f"Sent SIGTERM to process {pid}")
                        
                        # Terminate all children
                        for child in children:
                            try:
                                child.terminate()
                                logger.debug(f"Sent SIGTERM to child process {child.pid}")
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        logger.warning(f"Could not terminate {process_name} (PID: {pid})")
                    
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
                    
                    # Force kill if still running
                    if psutil.pid_exists(pid):
                        logger.warning(f"Force killing {process_name} (PID: {pid})")
                        try:
                            # Kill the main process
                            process.kill()
                            logger.debug(f"Sent SIGKILL to process {pid}")
                            
                            # Kill all children
                            for child in children:
                                try:
                                    child.kill()
                                    logger.debug(f"Sent SIGKILL to child process {child.pid}")
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            logger.warning(f"Could not kill {process_name} (PID: {pid})")
                        
                        # Final wait with shorter timeout
                        for _ in range(50):  # 5 seconds max
                            if not psutil.pid_exists(pid):
                                break
                            time.sleep(0.1)
                    
                    success = not psutil.pid_exists(pid)
                    if success:
                        logger.info(f"{process_name} terminated successfully")
                    else:
                        logger.error(f"Failed to terminate {process_name} (PID: {pid})")
                    return success
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    logger.info(f"Process {process_name} (PID: {pid}) no longer accessible")
                    return True
                except Exception as e:
                    logger.error(f"Error terminating {process_name} (PID: {pid}): {e}")
                    return False

            # Track termination success
            ai_stopped = terminate_process_group(pid, "AI service")
            api_stopped = terminate_process_group(app_pid, "API service")
            
            # Give processes time to clean up
            time.sleep(2)
            
            # Verify ports are freed with optimized checking
            ports_info = [(app_port, "API"), (local_ai_port, "AI")]
            ports_freed = self._check_ports_freed(ports_info)
            
            # Only clean up the pickle file if both services were successfully stopped
            pickle_removed = True
            if ai_stopped and api_stopped:
                try:
                    if os.path.exists(self.pickle_file):
                        os.remove(self.pickle_file)
                        logger.info("Service metadata file removed")
                    else:
                        logger.debug("Service metadata file already removed")
                except Exception as e:
                    logger.error(f"Error removing pickle file: {str(e)}")
                    pickle_removed = False
            else:
                logger.warning("Keeping service metadata file since not all processes were successfully stopped")
                pickle_removed = False

            # Determine overall success
            success = ai_stopped and api_stopped and pickle_removed
            
            if success:
                logger.info("AI service stopped successfully.")
                if not ports_freed:
                    logger.warning("Service stopped but some ports may still be in use temporarily")
            else:
                logger.error("AI service stop completed with some failures")
                logger.error(f"AI stopped: {ai_stopped}, API stopped: {api_stopped}, pickle removed: {pickle_removed}")
            
            return success

        except Exception as e:
            logger.error(f"Error stopping AI service: {str(e)}", exc_info=True)
            return False

    def _get_model_template_path(self, model_family: str) -> str:
        """Get the template path for a specific model family."""
        chat_template_path = pkg_resources.resource_filename("local_ai", f"examples/templates/{model_family}.jinja")
        # check if the template file exists
        if not os.path.exists(chat_template_path):
            return None
        return chat_template_path

    def _get_model_best_practice_path(self, model_family: str) -> str:
        """Get the best practices for a specific model family."""
        best_practice_path = pkg_resources.resource_filename("local_ai", f"examples/best_practices/{model_family}.json")
        # check if the best practices file exists
        if not os.path.exists(best_practice_path):
            return None
        return best_practice_path

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
            "--no-webui",
            "-ngl", "9999",
            "--no-mmap",
            "--mlock",
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

    def _load_or_fetch_metadata(self, hash: str, model_dir: str) -> tuple[str, dict]:
        """
        Load metadata from cache or fetch from remote with optimized I/O.
        Returns (folder_name, metadata_dict)
        """
        metadata_file = os.path.join(model_dir, f"{hash}.json")
        
        # Try to load from cache first
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    folder_name = metadata.get("folder_name", "")
                    logger.info(f"Loaded metadata from cache: {metadata_file}")
                    return folder_name, metadata
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Invalid metadata file {metadata_file}, will fetch from remote: {e}")
                # Remove corrupted file
                try:
                    os.remove(metadata_file)
                except OSError:
                    pass
        
        # Fetch from remote
        filecoin_url = f"https://gateway.lighthouse.storage/ipfs/{hash}"
        response_json = self._retry_request_json(filecoin_url, retries=3, delay=5, timeout=10)
        
        if not response_json:
            logger.error(f"Failed to fetch metadata from {filecoin_url}")
            return "", {}
        
        folder_name = response_json.get("folder_name", "")
        
        # Save to cache (create directory if needed)
        try:
            os.makedirs(model_dir, exist_ok=True)
            with open(metadata_file, "w") as f:
                json.dump(response_json, f)
            logger.info(f"Cached metadata to {metadata_file}")
        except IOError as e:
            logger.warning(f"Failed to cache metadata: {e}")
        
        return folder_name, response_json
    
    def _check_multimodal_support(self, local_model_path: str) -> tuple[bool, Optional[str]]:
        """
        Check if model supports multimodal with single file check.
        Returns (is_multimodal, projector_path)
        """
        projector_path = f"{local_model_path}-projector"
        is_multimodal = os.path.exists(projector_path)
        return is_multimodal, projector_path if is_multimodal else None

    def _create_service_metadata(self, hash: str, local_model_path: str, local_ai_port: int, 
                                port: int, context_length: int, metadata: dict, 
                                is_multimodal: bool, projector_path: Optional[str]) -> dict:
        """Create service metadata dictionary with all required fields."""
        return {
            "hash": hash,
            "port": local_ai_port,
            "local_text_path": local_model_path,
            "app_port": port,
            "context_length": context_length,
            "last_activity": time.time(),
            "multimodal": is_multimodal,
            "local_projector_path": projector_path,
            "family": metadata.get("family", "")
        }

    def _acquire_start_lock(self, hash: str, host: str, port: int) -> bool:
        """
        Acquire the start lock efficiently with minimal I/O operations.
        Returns True if lock acquired, False otherwise.
        """
        current_pid = os.getpid()
        current_time = time.time()
        
        # Check and handle existing lock
        if self.start_lock_file.exists():
            try:
                with open(self.start_lock_file, "r") as f:
                    lock_data = json.load(f)
                
                lock_pid = lock_data.get("pid")
                lock_timestamp = lock_data.get("timestamp", 0)
                
                # Check if lock is stale (over 30 minutes) or process is dead
                if current_time - lock_timestamp > self.LOCK_TIMEOUT or not psutil.pid_exists(lock_pid):
                    logger.warning(f"Removing stale lock file (PID: {lock_pid})")
                    self.start_lock_file.unlink()
                elif lock_pid != current_pid:
                    logger.error(f"Another process (PID: {lock_pid}) is already starting service")
                    return False
                    
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning(f"Corrupted lock file, removing: {e}")
                try:
                    self.start_lock_file.unlink()
                except OSError:
                    pass
        
        # Create new lock
        lock_data = {
            "pid": current_pid,
            "hash": hash,
            "timestamp": current_time,
            "host": host,
            "port": port
        }
        
        try:
            with open(self.start_lock_file, "w") as f:
                json.dump(lock_data, f)
            logger.info(f"Acquired start lock for PID {current_pid}")
            return True
        except OSError as e:
            logger.error(f"Failed to create lock file: {e}")
            return False
    
    def _release_start_lock(self) -> None:
        """Release the start lock if we own it."""
        if not self.start_lock_file.exists():
            return
            
        current_pid = os.getpid()
        try:
            with open(self.start_lock_file, "r") as f:
                lock_data = json.load(f)
            
            if lock_data.get("pid") == current_pid:
                self.start_lock_file.unlink()
                logger.info(f"Released start lock for PID {current_pid}")
            else:
                logger.warning(f"Lock file PID mismatch, not removing")
                
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"Error releasing lock: {e}")
            try:
                self.start_lock_file.unlink()
                logger.info("Removed potentially corrupted lock file")
            except OSError:
                pass

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