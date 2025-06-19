import os
import json
import time
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
from local_ai.download import download_model_from_filecoin_async

class LocalAIServiceError(Exception):
    """Base exception for Local AI service errors."""
    pass

class ServiceStartError(LocalAIServiceError):
    """Exception raised when service fails to start."""
    pass

class ServiceHealthError(LocalAIServiceError):
    """Exception raised when service health check fails."""
    pass

class ModelNotFoundError(LocalAIServiceError):
    """Exception raised when model file is not found."""
    pass

class LocalAIManager:
    """Manages a local AI service."""
    
    def __init__(self):
        """Initialize the LocalAIManager."""
        self.pickle_file = Path(os.getenv("RUNNING_SERVICE_FILE", "running_service.pkl"))
        # Stores information about all loaded models, keyed by model hash
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.llama_server_path = os.getenv("LLAMA_SERVER")
        if not self.llama_server_path or not os.path.exists(self.llama_server_path):
            raise LocalAIServiceError("llama-server executable not found in LLAMA_SERVER or PATH")
        
        # Load running services from pickle file if it exists
        if self.pickle_file.exists():
            try:
                with open(self.pickle_file, "rb") as f:
                    self.loaded_models = pickle.load(f)
                if not isinstance(self.loaded_models, dict): # Basic check for old format
                    logger.warning("Pickle file might be in old format. Initializing empty loaded_models.")
                    self.loaded_models = {}
                else:
                    logger.info(f"Loaded running services information for {len(self.loaded_models)} model(s) from {self.pickle_file}")
                    # After loading, try to update each model's FastAPI instance
                    # This is important if LocalAIManager restarts and FastAPI instances are still running.
                    for model_hash, service_metadata in self.loaded_models.items():
                        app_port = service_metadata.get("app_port")
                        model_name = service_metadata.get("model_name", model_hash)
                        if app_port:
                            try:
                                update_url = f"http://localhost:{app_port}/update"
                                # The FastAPI's /update endpoint expects the service metadata for that specific model
                                response = requests.post(update_url, json=service_metadata, timeout=5) # Short timeout
                                response.raise_for_status()
                                logger.info(f"Successfully sent /update to FastAPI instance for model '{model_name}' on port {app_port} after loading from pickle.")
                            except requests.exceptions.RequestException as re:
                                logger.warning(f"Failed to send /update to FastAPI instance for model '{model_name}' on port {app_port} after loading from pickle: {re}")
                        else:
                            logger.warning(f"No app_port found for model '{model_name}' in loaded pickle data. Cannot send /update.")
            except Exception as e:
                logger.error(f"Error loading running service from pickle: {str(e)}. Initializing empty loaded_models.")
                self.loaded_models = {}

    def _get_free_port(self) -> int:
        """Get a free port number."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def _wait_for_service(self, port: int, timeout: int = 300) -> bool:
        """
        Wait for the AI service to become healthy.

        Args:
            port (int): Port number of the service.
            timeout (int): Maximum time to wait in seconds (default: 300).

        Returns:
            bool: True if service is healthy, False otherwise.

        Raises:
            ServiceHealthError: If service fails to become healthy within timeout.
        """
        health_check_url = f"http://localhost:{port}/health"
        start_time = time.time()
        wait_time = 1  # Initial wait time in seconds
        last_error = None
        
        while time.time() - start_time < timeout:
            try:
                status = requests.get(health_check_url, timeout=5)
                if status.status_code == 200 and (status.json().get("status") == "ok"):
                    logger.debug(f"Service healthy at {health_check_url}")
                    return True
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.debug(f"Health check failed: {last_error}")
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, 60)  # Exponential backoff, max 60s
            
        raise ServiceHealthError(f"Service failed to become healthy within {timeout} seconds. Last error: {last_error}")
    
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
            # This restart logic needs to be updated for multi-model support.
            # For now, it's simplified or would need a specific hash to restart.
            # Placeholder:
            logger.warning("Restart functionality needs to be fully adapted for multi-model support.")
            # Example: stop all and start all previous ones if self.loaded_models was populated from pickle
            if not self.loaded_models: # Assuming loaded_models was populated if pickle existed
                 logger.warning("No models were previously loaded to restart.")
                 return False

            all_hashes_to_restart = list(self.loaded_models.keys())
            # Important: Need to store original ports and context lengths to restart them accurately.
            # This current restart will start them on new free ports.
            # A more robust restart would re-use previous ports if possible or specified.
            logger.info(f"Attempting to restart models: {all_hashes_to_restart}")
            self.stop() # Stop all current models

            # This simplistic restart won't preserve original ports unless start() is modified
            # or we pass specific port configurations.
            # For now, let's assume start will assign new ports.
            # We'd need to retrieve original settings for a true "restart".
            # This is a placeholder for a more robust restart implementation.
            # The `start` method will need the context_length for each model.
            # This information should be retrieved from `self.loaded_models` before calling `stop()`.
            # For now, this restart is non-functional for maintaining previous ports/contexts.
            # return self.start(all_hashes_to_restart, ???) # Needs context length and potentially ports
            logger.error("Full restart functionality for multiple models is not yet implemented.")
            return False
        except Exception as e:
            logger.error(f"Error in placeholder restart logic: {str(e)}", exc_info=True)
            return False

    def _get_family_template_and_practice(self, model_family: str):
        """Helper to get template and best practice paths based on folder name."""
        return (
            self._get_model_template_path(model_family),
            self._get_model_best_practice_path(model_family)
        )
    
    def _retry_request_json(self, url, retries=2, delay=2, timeout=8):
        """Utility to retry a GET request for JSON data with optimized parameters."""
        backoff_delay = delay
        for attempt in range(retries):
            try:
                response = requests.get(url, timeout=timeout)
                if response.status_code == 200:
                    # Cache JSON parsing result
                    json_data = response.json()
                    return json_data
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt+1} failed for {url}: {e}")
                if attempt < retries - 1:  # Don't sleep on last attempt
                    time.sleep(backoff_delay)
                    backoff_delay = min(backoff_delay * 1.5, 8)  # Exponential backoff with cap
        return None

    def start(self, model_hashes: list[str], host: str = "0.0.0.0", context_length: int = 32768, base_app_port: int = 11434) -> bool:
        """
        Start local AI services for the given model hashes.

        Args:
            model_hashes (list[str]): List of Filecoin hashes of the models to download and run.
            host (str): Host address for the AI services (default: "0.0.0.0").
            context_length (int): Context length for the models (default: 32768).
            base_app_port (int): Base port number for assigning to FastAPI apps. Ports will be incremented from here.

        Returns:
            bool: True if service started successfully, False otherwise.

        Raises:
            ValueError: If any model_hash is not provided.
            ModelNotFoundError: If a model file is not found.
            ServiceStartError: If any service fails to start.
        """
        if not model_hashes:
            raise ValueError("At least one Filecoin hash is required to start services.")

        models_started_successfully = 0
        current_app_port = base_app_port

        for model_hash in model_hashes:
            if not model_hash:
                logger.warning("Empty model hash provided, skipping.")
                continue

            if model_hash in self.loaded_models:
                logger.warning(f"Model with hash '{model_hash}' is already loaded or running. Skipping.")
                # Consider if we should check its health or allow restart here.
                # For now, just increment if it's healthy, otherwise it might need cleanup.
                # This part needs more robust handling for "already running" cases.
                # if self._is_service_healthy(self.loaded_models[model_hash].get("app_port")):
                #    models_started_successfully +=1
                continue

            logger.info(f"Starting local AI service for model with hash: {model_hash}")
            
            try:
                local_model_path = asyncio.run(download_model_from_filecoin_async(model_hash))
                local_projector_path = local_model_path + "-projector"

                if not os.path.exists(local_model_path):
                    raise ModelNotFoundError(f"Model file not found at: {local_model_path} for hash {model_hash}")

                llama_server_port = self._get_free_port()
                # Ensure fast_api_port is different and manages collisions
                fast_api_port = self._get_free_port()
                while fast_api_port == llama_server_port: # Ensure different ports
                    fast_api_port = self._get_free_port()

                # In a multi-model setup, current_app_port should be unique for each FastAPI app.
                # The CLI start command might need to pass a base_app_port or ports list.
                # For now, let's use _get_free_port for the app_port as well.
                app_port_for_this_model = fast_api_port # Using the new free port for this app

                service_metadata_for_model = {
                    "hash": model_hash,
                    "llama_server_port": llama_server_port,
                    "local_model_path": local_model_path,
                    "app_port": app_port_for_this_model,
                    "context_length": context_length, # This might need to be model-specific
                    "last_activity": time.time(),
                    "multimodal": os.path.exists(local_projector_path),
                    "local_projector_path": local_projector_path if os.path.exists(local_projector_path) else None
                }

                model_dir = os.path.dirname(local_model_path)
                metadata_file_path = os.path.join(model_dir, f"{model_hash}.json")
                logger.info(f"metadata_file: {metadata_file_path}")
                folder_name = ""

                if os.path.exists(metadata_file_path):
                    try:
                        with open(metadata_file_path, "r") as f:
                            metadata = json.load(f)
                            service_metadata_for_model["family"] = metadata.get("family", "")
                            folder_name = metadata.get("folder_name", model_hash) # Use hash if no folder_name
                            service_metadata_for_model["model_name"] = folder_name
                            logger.info(f"Loaded metadata from {metadata_file_path}")
                    except Exception as e:
                        logger.error(f"Error loading metadata file {metadata_file_path}: {e}")
                        # Fallback to default name if metadata read fails
                        service_metadata_for_model["model_name"] = model_hash
                else:
                    # Fetch from Filecoin URL (assuming this returns necessary details)
                    filecoin_url = f"https://gateway.lighthouse.storage/ipfs/{model_hash}"
                    response_json = self._retry_request_json(filecoin_url, retries=3, delay=5, timeout=10)
                    if response_json:
                        folder_name = response_json.get("folder_name", model_hash)
                        service_metadata_for_model["family"] = response_json.get("family", "")
                        service_metadata_for_model["model_name"] = folder_name
                        try:
                            with open(metadata_file_path, "w") as f:
                                json.dump(response_json, f)
                            logger.info(f"Saved metadata to {metadata_file_path}")
                        except Exception as e:
                            logger.error(f"Error saving metadata file {metadata_file_path}: {e}")
                    else:
                        logger.warning(f"Could not fetch metadata for {model_hash} from {filecoin_url}. Using defaults.")
                        service_metadata_for_model["model_name"] = model_hash


                # Adjust context length for specific models if needed (example from original code)
                current_model_context_length = context_length
                if "gemma" in folder_name.lower():
                    template_path, best_practice_path = self._get_family_template_and_practice("gemma")
                    current_model_context_length = context_length // 2 # Gemma specific adjustment
                    running_ai_command = self._build_ai_command(
                        local_model_path, llama_server_port, host, current_model_context_length, template_path
                    )
                # Add other model specific conditions here (qwen25, qwen3, llama)
                elif "qwen25" in folder_name.lower():
                    template_path, best_practice_path = self._get_family_template_and_practice("qwen25")
                    running_ai_command = self._build_ai_command(
                        local_model_path, llama_server_port, host, current_model_context_length, template_path, best_practice_path
                    )
                elif "qwen3" in folder_name.lower():
                    template_path, best_practice_path = self._get_family_template_and_practice("qwen3")
                    running_ai_command = self._build_ai_command(
                        local_model_path, llama_server_port, host, current_model_context_length, template_path, best_practice_path
                    )
                elif "llama" in folder_name.lower(): # Assuming 'llama' is a general family name
                    template_path, best_practice_path = self._get_family_template_and_practice("llama")
                    running_ai_command = self._build_ai_command(
                        local_model_path, llama_server_port, host, current_model_context_length, template_path, best_practice_path
                    )
                else:
                    running_ai_command = self._build_ai_command(
                        local_model_path, llama_server_port, host, current_model_context_length
                    )
                service_metadata_for_model["context_length"] = current_model_context_length


                if service_metadata_for_model["multimodal"]:
                    running_ai_command.extend(["--mmproj", str(local_projector_path)])

                logger.info(f"Starting llama-server for {model_hash}: {' '.join(map(str,running_ai_command))}")
                service_metadata_for_model["running_ai_command"] = running_ai_command

                os.makedirs("logs", exist_ok=True)
                ai_log_stderr_path = Path(f"logs/ai_{model_hash}_{llama_server_port}.log")
                ai_process = None
                try:
                    with open(ai_log_stderr_path, 'w') as stderr_log:
                        ai_process = subprocess.Popen(
                            running_ai_command, stderr=stderr_log, preexec_fn=os.setsid
                        )
                    logger.info(f"llama-server for {model_hash} logs at {ai_log_stderr_path}")
                except Exception as e:
                    logger.error(f"Error starting llama-server for {model_hash}: {str(e)}", exc_info=True)
                    continue # Next model

                if not self._wait_for_service(llama_server_port):
                    logger.error(f"llama-server for {model_hash} failed to start on port {llama_server_port}.")
                    if ai_process: ai_process.terminate()
                    continue # Next model

                service_metadata_for_model["pid"] = ai_process.pid

                # Start the FastAPI app for this model
                # The FastAPI app (local_ai.apis:app) needs to be aware of which llama-server instance to talk to.
                # This is typically done by passing configuration to it.
                # The existing /update mechanism seems designed for a single global service.
                # This will require significant changes to how local_ai.apis:app is structured or configured.
                # For now, let's assume each FastAPI app can be configured via an /update call on its unique port.

                uvicorn_command = [
                    "uvicorn", "local_ai.apis:app", "--host", host,
                    "--port", str(app_port_for_this_model), "--log-level", "info"
                ]
                logger.info(f"Starting FastAPI app for {model_hash}: {' '.join(uvicorn_command)}")
                api_log_stderr_path = Path(f"logs/api_{model_hash}_{app_port_for_this_model}.log")
                apis_process = None
                try:
                    with open(api_log_stderr_path, 'w') as stderr_log:
                        apis_process = subprocess.Popen(
                            uvicorn_command, stderr=stderr_log, preexec_fn=os.setsid
                        )
                    logger.info(f"FastAPI for {model_hash} logs at {api_log_stderr_path}")
                except Exception as e:
                    logger.error(f"Error starting FastAPI for {model_hash}: {str(e)}", exc_info=True)
                    if ai_process: ai_process.terminate() # Clean up llama-server
                    continue # Next model

                if not self._wait_for_service(app_port_for_this_model): # Wait for FastAPI app
                    logger.error(f"FastAPI for {model_hash} failed to start on port {app_port_for_this_model}.")
                    if ai_process: ai_process.terminate()
                    if apis_process: apis_process.terminate()
                    continue # Next model

                service_metadata_for_model["app_pid"] = apis_process.pid

                # Update this specific FastAPI app instance
                try:
                    update_url = f"http://localhost:{app_port_for_this_model}/update"
                    # The metadata sent to /update should be specific to this service
                    # e.g., it needs to know its own llama_server_port, model_hash etc.
                    # The current service_metadata_for_model contains this.
                    response = requests.post(update_url, json=service_metadata_for_model, timeout=10)
                    response.raise_for_status()
                    logger.info(f"FastAPI instance for {model_hash} on port {app_port_for_this_model} updated.")
                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to update FastAPI instance for {model_hash} on port {app_port_for_this_model}: {str(e)}")
                    if ai_process: ai_process.terminate()
                    if apis_process: apis_process.terminate()
                    continue # Next model

                self.loaded_models[model_hash] = service_metadata_for_model
                models_started_successfully += 1
                logger.info(f"Service for model {model_hash} started successfully. llama-server on {llama_server_port}, FastAPI on {app_port_for_this_model}.")

            except ModelNotFoundError as e:
                logger.error(f"Model not found for hash {model_hash}: {str(e)}")
            except ServiceStartError as e:
                logger.error(f"Service start error for hash {model_hash}: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error starting service for hash {model_hash}: {str(e)}", exc_info=True)

        if models_started_successfully > 0:
            self._dump_running_service() # Save all successfully started models
            return True
        
        return False # No models started

    def _dump_running_service(self):
        """Dump the self.loaded_models dictionary to a file."""
        try:
            with open(self.pickle_file, "wb") as f:
                pickle.dump(self.loaded_models, f)
            logger.info(f"Running services state saved to {self.pickle_file}")
        except Exception as e:
            logger.error(f"Error dumping running services state: {str(e)}", exc_info=True)
            # No return False here, as the calling function might not expect it.

    def get_running_model(self) -> Optional[str]:
        """
        Get details of the first running model, if any.
        (As per subtask: "return the details of the first model in self.loaded_models if any, or None")

        Returns:
            Optional[Dict[str, Any]]: Details of the first loaded model, or None.
        """
        if not self.loaded_models:
            # Attempt to load from pickle if not already loaded (e.g. new manager instance)
            if self.pickle_file.exists():
                try:
                    with open(self.pickle_file, "rb") as f:
                        loaded = pickle.load(f)
                        # Basic check if it's the new format
                        if isinstance(loaded, dict) and all(isinstance(k, str) and isinstance(v, dict) for k, v in loaded.items()):
                             self.loaded_models = loaded
                        else: # Old format or unexpected structure
                            # If it's the old single model format, we might try to adapt it or log a warning.
                            # For now, assume it's new format or invalid.
                            logger.warning("Pickle file data is not in the expected multi-model format. Ignoring.")
                            self.loaded_models = {}
                except Exception as e:
                    logger.error(f"Error loading pickle in get_running_model: {e}")
                    self.loaded_models = {} # Ensure it's a dict

            if not self.loaded_models:
                return None

        # Return details of the "first" model. Dicts are unordered before 3.7,
        # but generally iterate in insertion order in CPython 3.6+
        # For true "first" per original insertion, this is okay.
        try:
            first_model_hash = next(iter(self.loaded_models))
            return self.loaded_models[first_model_hash]
        except StopIteration: # Should not happen if self.loaded_models was checked
            return None
        except Exception as e: # General catch for safety
            logger.error(f"Error retrieving first running model: {e}")
            return None

    def stop(self) -> bool:
        """
        Stop all running AI services.

        Returns:
            bool: True if all services stopped successfully, False otherwise.
        """
        if not self.loaded_models:
            # Also check pickle file in case manager was just initialized
            if self.pickle_file.exists():
                try:
                    with open(self.pickle_file, "rb") as f:
                        # Ensure what's loaded is compatible with self.loaded_models structure
                        potential_loaded_models = pickle.load(f)
                        if isinstance(potential_loaded_models, dict) and \
                           all(isinstance(v, dict) for v in potential_loaded_models.values()):
                            self.loaded_models = potential_loaded_models
                        else: # Old format or unexpected
                            logger.warning("Pickle file is in an old or unexpected format. Cannot stop services based on it.")
                            # If it's old format, we might try to stop that single service.
                            # For now, we only act on the new multi-model format.
                except Exception as e:
                    logger.error(f"Error loading pickle file in stop(): {e}")
            
            if not self.loaded_models:
                logger.warning("No running AI services to stop.")
                # Ensure pickle file is removed if it exists but is empty/invalid for new format
                if self.pickle_file.exists():
                    try:
                        os.remove(self.pickle_file)
                    except OSError as e:
                        logger.error(f"Error removing stale pickle file: {e}")
                return False # Or True, as there's nothing to stop. Let's say False as no action taken.

        logger.info(f"Stopping all {len(self.loaded_models)} AI services...")
        all_stopped_successfully = True

        # Iterate over a copy of items for safe removal if needed, though clear() is used later
        models_to_stop = list(self.loaded_models.items())

        for model_hash, service_info in models_to_stop:
            pid = service_info.get("pid")
            app_pid = service_info.get("app_pid")
            model_name = service_info.get("model_name", model_hash) # Use model_name or hash for logging
            app_port = service_info.get("app_port") # For logging

            logger.info(f"Stopping service for model '{model_name}' (Hash: {model_hash}) on App Port {app_port} (AI PID: {pid}, API PID: {app_pid})...")

            def terminate_process_group(proc_pid, process_name_suffix, timeout=15):
                """
                Terminate a process and its entire process group.
                Returns True if successful, False otherwise.
                """
                # Ensure pid is used from the argument, not the outer scope one
                process_name = f"{model_name} {process_name_suffix}"
                current_pid = proc_pid

                if not current_pid:
                    logger.warning(f"No PID provided for {process_name}")
                    return True
                
                import signal # Should be at module level, but keeping original structure for now
                    
                try:
                    if not psutil.pid_exists(current_pid):
                        logger.info(f"Process {process_name} (PID: {current_pid}) not found, assuming already stopped")
                        return True
                    
                    try:
                        process = psutil.Process(current_pid)
                        status = process.status()
                        if status == psutil.STATUS_ZOMBIE:
                            logger.info(f"Process {process_name} (PID: {current_pid}) is already zombie, cleaning up")
                            return True
                        elif status in [psutil.STATUS_DEAD, psutil.STATUS_STOPPED]:
                            logger.info(f"Process {process_name} (PID: {current_pid}) is already dead/stopped")
                            return True
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        logger.info(f"Process {process_name} (PID: {current_pid}) no longer accessible, assuming stopped")
                        return True
                    
                    logger.info(f"Terminating {process_name} (PID: {current_pid})...")
                    
                    success = False
                    try:
                        children = []
                        try:
                            parent = psutil.Process(current_pid)
                            children = parent.children(recursive=True)
                            logger.debug(f"Found {len(children)} child processes for {process_name}")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                        
                        try:
                            pgid = os.getpgid(current_pid)
                            logger.debug(f"Sending SIGTERM to process group {pgid} for {process_name}")
                            os.killpg(pgid, signal.SIGTERM)
                        except (ProcessLookupError, OSError, PermissionError) as e:
                            logger.debug(f"Process group termination failed for {process_name}: {e}. Falling back.")
                            try:
                                parent = psutil.Process(current_pid)
                                parent.terminate()
                                for child in children:
                                    try: child.terminate()
                                    except (psutil.NoSuchProcess, psutil.AccessDenied): pass
                            except (psutil.NoSuchProcess, psutil.AccessDenied) as e_ind:
                                logger.warning(f"Could not terminate {process_name} (PID: {current_pid}): {e_ind}")
                                return True

                        start_time = time.time()
                        poll_interval = 0.1
                        while time.time() - start_time < timeout:
                            if not psutil.pid_exists(current_pid):
                                success = True; break
                            try:
                                process = psutil.Process(current_pid)
                                if process.status() == psutil.STATUS_ZOMBIE:
                                    success = True; break
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                success = True; break
                            time.sleep(poll_interval)
                            poll_interval = min(poll_interval * 1.2, 0.5)
                        if success: logger.info(f"{process_name} terminated gracefully")
                            
                    except Exception as e:
                        logger.error(f"Error during graceful termination of {process_name}: {e}")
                    
                    if not success and psutil.pid_exists(current_pid):
                        logger.warning(f"{process_name} (PID: {current_pid}) still running, sending SIGKILL")
                        try:
                            children = []
                            try:
                                parent = psutil.Process(current_pid)
                                children = parent.children(recursive=True)
                            except (psutil.NoSuchProcess, psutil.AccessDenied): pass

                            try:
                                pgid = os.getpgid(current_pid)
                                os.killpg(pgid, signal.SIGKILL)
                            except (ProcessLookupError, OSError, PermissionError):
                                try:
                                    parent = psutil.Process(current_pid)
                                    parent.kill()
                                    for child in children:
                                        try: child.kill()
                                        except (psutil.NoSuchProcess, psutil.AccessDenied): pass
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    logger.info(f"{process_name} disappeared during force kill")
                                    return True
                            
                            force_timeout = min(timeout // 2, 10)
                            start_time = time.time()
                            while time.time() - start_time < force_timeout:
                                if not psutil.pid_exists(current_pid):
                                    success = True; break
                                time.sleep(0.2)
                            
                            if psutil.pid_exists(current_pid):
                                try:
                                    process = psutil.Process(current_pid)
                                    if process.status() == psutil.STATUS_ZOMBIE: success = True
                                    else: logger.error(f"Failed to kill {process_name} (PID: {current_pid}), status: {process.status()}")
                                except (psutil.NoSuchProcess, psutil.AccessDenied): success = True
                            else: success = True
                            if success : logger.info(f"{process_name} force-killed successfully or disappeared")

                        except Exception as e:
                            logger.error(f"Error force-killing {process_name} (PID: {current_pid}): {str(e)}")
                
                except Exception as e: # Outer try for terminate_process_group
                    logger.error(f"Unexpected error terminating {process_name} (PID: {current_pid}): {str(e)}")
                return success # Return status of termination

            ai_stopped = terminate_process_group(pid, "llama-server")
            api_stopped = terminate_process_group(app_pid, "FastAPI app")

            if not (ai_stopped and api_stopped):
                all_stopped_successfully = False
                logger.error(f"Failed to completely stop service for model '{model_name}' (Hash: {model_hash}). AI stopped: {ai_stopped}, API stopped: {api_stopped}")
            else:
                logger.info(f"Service for model '{model_name}' (Hash: {model_hash}) stopped.")

        # After attempting to stop all, clear internal state and pickle file
        self.loaded_models.clear()
        pickle_removed = True
        if os.path.exists(self.pickle_file):
            try:
                os.remove(self.pickle_file)
                logger.info("All services stopped. Service metadata file removed.")
            except Exception as e:
                logger.error(f"Error removing pickle file after stopping all services: {str(e)}")
                pickle_removed = False
                all_stopped_successfully = False # Consider this a failure in cleanup

        # The port freeing check from original code can be complex with multiple ports.
        # For now, assume OS handles port freeing adequately once processes are terminated.
        # A more robust check could be added later if needed.

        if all_stopped_successfully:
            logger.info("All AI services stopped successfully.")
        else:
            logger.error("Some AI services may not have stopped cleanly.")
            
        return all_stopped_successfully

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
            "--pooling", "mean",
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