"""
This module provides a FastAPI application that acts as a proxy or processor for chat completion and embedding requests,
forwarding them to an underlying service running on a local port. It handles both text and vision-based chat completions,
as well as embedding generation, with support for streaming responses.
"""

import os
import logging
import httpx
import asyncio
import time
import json
import uuid
import subprocess
import signal
import requests
import psutil
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

# Import schemas from schema.py
from local_ai.schema import (
    Choice,
    Message,
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse
)

# Set up logging with both console and file output
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI()

# Constants for dynamic unload feature - Optimized for performance
IDLE_TIMEOUT = 600  # 10 minutes in seconds
UNLOAD_CHECK_INTERVAL = 30  # Check every 30 seconds (reduced from 60)
SERVICE_START_TIMEOUT = 45  # Reduced timeout for faster failure detection
POOL_CONNECTIONS = 50  # Further optimized for better resource usage
POOL_KEEPALIVE = 10  # Reduced keepalive to free connections faster
HTTP_TIMEOUT = 180.0  # Reduced timeout for faster failure detection
STREAM_TIMEOUT = 300.0  # Keep streaming timeout longer for large responses
MAX_RETRIES = 2  # Reduced retries for faster failure handling
RETRY_DELAY = 0.5  # Reduced delay between retries
MAX_QUEUE_SIZE = 50  # Further reduced for better memory usage
HEALTH_CHECK_INTERVAL = 0.5  # Faster health checks
STREAM_CHUNK_SIZE = 8  # User-specified chunk size for minimal initial buffering

# Utility functions
def get_this_instance_model_service_info() -> Dict[str, Any]:
    """
    Get service info for the model this FastAPI instance serves.
    Assumes app.state.service_info directly holds the metadata for this one model.
    """
    if not hasattr(app.state, "service_info") or not app.state.service_info:
        # This means LocalAIManager hasn't called /update on this instance yet.
        raise HTTPException(status_code=503, detail="Service information not yet populated for this model instance.")

    # app.state.service_info is expected to be a Dict[str, Any] (the model's own metadata)
    if not isinstance(app.state.service_info, dict) or not app.state.service_info.get("model_name"):
         # This would indicate malformed data sent by LocalAIManager or incorrect state.
         raise HTTPException(status_code=503, detail="Service information is malformed or incomplete for this model instance.")
    return app.state.service_info

def get_this_instance_model_llama_server_port() -> int:
    """Get the llama_server_port for the model this FastAPI instance serves."""
    model_service_info = get_this_instance_model_service_info()
    # 'llama_server_port' is the key used in LocalAIManager for the actual AI service port
    port = model_service_info.get("llama_server_port")
    if not port or not isinstance(port, int): # Ensure port is an int
        raise HTTPException(status_code=503, detail="Llama server port not configured or invalid for this model instance.")
    return port

def convert_request_to_dict(request) -> Dict[str, Any]:
    """Convert request object to dictionary, supporting both Pydantic v1 and v2."""
    return request.model_dump() if hasattr(request, "model_dump") else request.dict()

def generate_request_id() -> str:
    """Generate a short request ID for tracking."""
    return str(uuid.uuid4())[:8]

def generate_chat_completion_id() -> str:
    """Generate a chat completion ID."""
    return f"chatcmpl-{uuid.uuid4().hex}"

# Service Functions
class ServiceHandler:
    """Handler class for making requests to the underlying service."""
    
    @staticmethod
    async def kill_ai_server(model_id: Optional[str] = None) -> bool:
        """
        Kill the AI server process for the model served by this FastAPI instance.
        If model_id is provided, it's used for assertion.
        """
        service_info_to_use = get_this_instance_model_service_info()
        instance_model_name = service_info_to_use.get("model_name", "this_instance")

        if model_id and model_id != instance_model_name:
            logger.error(f"kill_ai_server called with model_id '{model_id}' which does not match this instance's model '{instance_model_name}'. Aborting kill.")
            # In a strict one-model-per-instance, this is an error.
            # Depending on desired behavior, could raise HTTPException or just return False.
            return False # Or raise HTTPException(status_code=400, ...)

        log_model_name = instance_model_name # Use the instance's actual model name for logging
        pid = service_info_to_use.get("pid") # Get PID from this instance's model info

        logger.info(f"Attempting to kill AI server for model '{log_model_name}' (this instance) with PID {pid}")
        if not pid:
            logger.warning(f"No PID found in service info for model '{log_model_name}' (this instance), cannot kill AI server.")
            return False
            
        # pid = service_info["pid"] # This line was a bug in previous version, ensure it's removed or corrected.
        # Corrected: pid is already fetched from service_info_to_use.
            logger.info(f"Attempting to kill AI server with PID {pid}")
            
            # Check if process exists and get its status
            if not psutil.pid_exists(pid):
                logger.info(f"Process {pid} not found, assuming already stopped")
                service_info.pop("pid", None)
                return True
            
            # Get process object and check its status
            try:
                process = psutil.Process(pid)
                status = process.status()
                if status == psutil.STATUS_ZOMBIE:
                    logger.info(f"Process {pid} is already zombie, cleaning up")
                    service_info.pop("pid", None)
                    return True
                elif status in [psutil.STATUS_DEAD, psutil.STATUS_STOPPED]:
                    logger.info(f"Process {pid} is already dead/stopped")
                    service_info.pop("pid", None)
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                logger.info(f"Process {pid} no longer accessible, assuming stopped")
                service_info.pop("pid", None)
                return True
            
            # First attempt: Graceful termination
            success = False
            timeout = 15  # Total timeout for graceful termination
            
            try:
                # Try to get all child processes first
                children = []
                try:
                    parent = psutil.Process(pid)
                    children = parent.children(recursive=True)
                    logger.debug(f"Found {len(children)} child processes for AI server")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                
                # Try process group termination first
                try:
                    pgid = os.getpgid(pid)
                    logger.debug(f"Sending SIGTERM to process group {pgid}")
                    os.killpg(pgid, signal.SIGTERM)
                    logger.info(f"Successfully sent SIGTERM to process group {pgid}")
                except (ProcessLookupError, OSError, PermissionError) as e:
                    logger.debug(f"Process group termination failed: {e}")
                    # Fall back to individual process termination
                    try:
                        parent = psutil.Process(pid)
                        parent.terminate()
                        logger.info(f"Successfully sent SIGTERM to process {pid}")
                        
                        # Also terminate children individually if group kill failed
                        for child in children:
                            try:
                                child.terminate()
                                logger.debug(f"Terminated child process {child.pid}")
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        logger.warning(f"Could not terminate process {pid}: {e}")
                        service_info.pop("pid", None)
                        return True  # Process might already be gone

                # Wait for graceful termination
                start_time = time.time()
                check_interval = 0.5
                while time.time() - start_time < timeout:
                    try:
                        if not psutil.pid_exists(pid):
                            logger.info(f"AI server process terminated gracefully")
                            success = True
                            break
                        
                        # Check if it's a zombie that needs cleanup
                        try:
                            process = psutil.Process(pid)
                            if process.status() == psutil.STATUS_ZOMBIE:
                                logger.info(f"AI server became zombie, considering it stopped")
                                success = True
                                break
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            success = True
                            break
                            
                        await asyncio.sleep(check_interval)
                    except Exception:
                        # If we can't check, assume it's gone
                        success = True
                        break
                
                if success:
                    service_info.pop("pid", None)
                    return True
                    
            except Exception as e:
                logger.error(f"Error during graceful termination: {e}")
            
            # Second attempt: Force termination
            if not success and psutil.pid_exists(pid):
                logger.warning(f"Process {pid} still running after SIGTERM, sending SIGKILL")
                try:
                    # Get fresh list of children
                    children = []
                    try:
                        parent = psutil.Process(pid)
                        children = parent.children(recursive=True)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    
                    # Try process group kill first
                    try:
                        pgid = os.getpgid(pid)
                        os.killpg(pgid, signal.SIGKILL)
                        logger.info(f"Successfully sent SIGKILL to process group {pgid}")
                    except (ProcessLookupError, OSError, PermissionError):
                        # Fall back to individual process kill
                        try:
                            parent = psutil.Process(pid)
                            parent.kill()
                            logger.info(f"Successfully sent SIGKILL to process {pid}")
                            
                            # Kill children individually
                            for child in children:
                                try:
                                    child.kill()
                                    logger.debug(f"Killed child process {child.pid}")
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            logger.info(f"Process disappeared during force kill")
                            service_info.pop("pid", None)
                            return True
                    
                    # Wait for force termination
                    force_timeout = 10  # Max 10 seconds for force kill
                    start_time = time.time()
                    while time.time() - start_time < force_timeout:
                        if not psutil.pid_exists(pid):
                            logger.info(f"AI server killed successfully")
                            success = True
                            break
                        await asyncio.sleep(0.2)
                    
                    # Final check
                    if psutil.pid_exists(pid):
                        try:
                            process = psutil.Process(pid)
                            status = process.status()
                            if status == psutil.STATUS_ZOMBIE:
                                logger.warning(f"AI server is zombie but considered stopped")
                                success = True
                            else:
                                logger.error(f"Failed to kill AI server (PID: {pid}), status: {status}")
                                return False
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            success = True
                    else:
                        success = True
                        
                except Exception as e:
                    logger.error(f"Error force-killing AI server (PID: {pid}): {str(e)}")
                    return False
            
            # Clean up service info for this instance's model
            if success:
                # service_info_to_use is app.state.service_info for this instance
                app.state.service_info.pop("pid", None)
                logger.info(f"AI server for model '{log_model_name}' (this instance) stopped successfully.")
                return True
            else:
                logger.error(f"Failed to stop AI server for model '{log_model_name}'.")
                return False
            
        except Exception as e:
            logger.error(f"Error killing AI server: {str(e)}", exc_info=True)
            return False
    
    @staticmethod
    async def reload_ai_server(model_id: Optional[str] = None, service_start_timeout: int = SERVICE_START_TIMEOUT) -> bool:
        """
        Reload the AI server process for the model served by this FastAPI instance.
        If model_id is provided, it's used for assertion.
        """
        service_info_to_reload = get_this_instance_model_service_info()
        instance_model_name = service_info_to_reload.get("model_name", "this_instance")

        if model_id and model_id != instance_model_name:
            logger.error(f"reload_ai_server called with model_id '{model_id}' which does not match instance model '{instance_model_name}'. Aborting reload.")
            return False # Or raise HTTPException

        log_model_name = instance_model_name
        # original_model_hash is not strictly needed here if app.state.service_info is this instance's direct state

        if "running_ai_command" not in service_info_to_reload:
            logger.error(f"No running_ai_command found for model '{log_model_name}' (this instance), cannot reload.")
                return False
                
            running_ai_command = service_info_to_reload["running_ai_command"]
            logger.info(f"Reloading AI server for model '{log_model_name}' with command: {running_ai_command}")

            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            # Log file should be specific to the model instance
            llama_server_port_for_log = service_info_to_reload.get("llama_server_port", "unknownport")
            ai_log_stderr = logs_dir / f"ai_{log_model_name.replace('/', '_')}_{llama_server_port_for_log}.log"
            
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
            
            # Wait for the process to start by checking the health endpoint
            llama_server_port = service_info_to_reload.get("llama_server_port")
            if not llama_server_port:
                logger.error(f"Cannot check health for model '{log_model_name}': llama_server_port not found.")
                return False

            start_time = time.time()
            
            while time.time() - start_time < service_start_timeout:
                try:
                    response = requests.get(f"http://localhost:{llama_server_port}/health", timeout=2)
                    if response.status_code == 200:
                        logger.info(f"Service health check passed for model '{log_model_name}' after {time.time() - start_time:.2f}s")
                        break
                except (requests.RequestException, ConnectionError):
                    await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            
            # Check if the process is running
            if ai_process.poll() is None:
                # Update PID in this instance's app.state.service_info
                app.state.service_info["pid"] = ai_process.pid
                # The hash isn't strictly necessary for logging if we always refer to "this instance"
                instance_hash = service_info_to_reload.get("hash", "unknown_hash")
                logger.info(f"Successfully reloaded AI server for model '{log_model_name}' (hash: {instance_hash}, this instance) with new PID {ai_process.pid}.")
                return True
            else:
                logger.error(f"Failed to reload AI server: Process exited with code {ai_process.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"Error reloading AI server: {str(e)}", exc_info=True)
            return False
    
    @staticmethod
    def _create_vision_error_response(request: ChatCompletionRequest, content: str):
        """Create error response for vision requests when multimodal is not supported."""
        if request.stream:
            async def error_stream():
                chunk = {
                    "id": generate_chat_completion_id(),
                    "choices": [{
                        "delta": {"content": content},
                        "finish_reason": "stop",
                        "index": 0
                    }],
                    "created": int(time.time()),
                    "model": request.model,
                    "object": "chat.completion.chunk"
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream")
        else:
            return ChatCompletionResponse(
                id=generate_chat_completion_id(),
                object="chat.completion",
                created=int(time.time()),
                model=request.model,
                choices=[Choice(
                    finish_reason="stop",
                    index=0,
                    message=Message(role="assistant", content=content)
                )]
            )
    
    @staticmethod
    async def generate_text_response(request: ChatCompletionRequest, model_id: str):
        """Generate a response for chat completion requests, supporting both streaming and non-streaming."""
        # model_id is the user-facing model name. This FastAPI instance serves ONE model.
        # Ensure the requested model_id matches the one this instance serves.
        this_instance_service_info = get_this_instance_model_service_info()
        this_instance_model_name = this_instance_service_info.get("model_name")

        if model_id != this_instance_model_name:
            logger.error(f"Request for model '{model_id}' reached instance serving '{this_instance_model_name}'. Mismatch.")
            raise HTTPException(status_code=400, detail=f"Model '{model_id}' not served by this API instance (serves '{this_instance_model_name}').")

        llama_server_port = get_this_instance_model_llama_server_port() # Gets port for this instance's model
        
        if request.is_vision_request():
            if not this_instance_service_info.get("multimodal", False): # Check this instance's capability
                content = f"Model '{this_instance_model_name}' (this instance) is not equipped to interpret images."
                return ServiceHandler._create_vision_error_response(request, content)
                
        request.clean_messages()
        request.enhance_tool_messages()
        request_dict = convert_request_to_dict(request)
        
        if request.stream:
            return StreamingResponse(
                ServiceHandler._stream_generator(llama_server_port, request_dict), # Pass specific port
                media_type="text/event-stream"
            )

        # Make a non-streaming API call
        response_data = await ServiceHandler._make_api_call(llama_server_port, "/v1/chat/completions", request_dict) # Pass specific port
        return ChatCompletionResponse(
            id=response_data.get("id", generate_chat_completion_id()),
            object=response_data.get("object", "chat.completion"),
            created=response_data.get("created", int(time.time())),
            model=request.model,
            choices=response_data.get("choices", [])
        )
    
    @staticmethod
    async def generate_embeddings_response(request: EmbeddingRequest, model_id: str):
        """Generate a response for embedding requests."""
        # Similar to generate_text_response, ensure model_id matches this instance.
        this_instance_service_info = get_this_instance_model_service_info()
        this_instance_model_name = this_instance_service_info.get("model_name")

        if model_id != this_instance_model_name:
            logger.error(f"Request for embedding model '{model_id}' reached instance serving '{this_instance_model_name}'. Mismatch.")
            raise HTTPException(status_code=400, detail=f"Embedding model '{model_id}' not served by this API instance (serves '{this_instance_model_name}').")

        llama_server_port = get_this_instance_model_llama_server_port()
        request_dict = convert_request_to_dict(request)
        response_data = await ServiceHandler._make_api_call(llama_server_port, "/v1/embeddings", request_dict)
        return EmbeddingResponse(
            object=response_data.get("object", "list"),
            data=response_data.get("data", []),
            model=request.model
        )
    
    @staticmethod
    async def _make_api_call(port: int, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a non-streaming API call to the specified endpoint and return the JSON response."""
        try:
            response = await app.state.client.post(
                f"http://localhost:{port}{endpoint}", 
                json=data,
                timeout=HTTP_TIMEOUT
            )
            logger.info(f"Received response with status code: {response.status_code}")
            
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"Error: {response.status_code} - {error_text}")
                if response.status_code < 500:
                    raise HTTPException(status_code=response.status_code, detail=error_text)
            
            # Cache JSON parsing to avoid multiple calls
            json_response = response.json()
            return json_response
            
        except httpx.TimeoutException as e:
            raise HTTPException(status_code=504, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @staticmethod
    async def _stream_generator(port: int, data: Dict[str, Any]):
        """Generator for streaming responses from the service."""
        try:
            async with app.state.client.stream(
                "POST", 
                f"http://localhost:{port}/v1/chat/completions", 
                json=data,
                timeout=STREAM_TIMEOUT
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    error_msg = f"data: {{\"error\":{{\"message\":\"{error_text.decode()}\",\"code\":{response.status_code}}}}}\n\n"
                    logger.error(f"Streaming error: {response.status_code} - {error_text.decode()}")
                    yield error_msg
                    return
                
                buffer = bytearray()  # Use bytearray for more efficient byte operations
                async for chunk in response.aiter_bytes(chunk_size=STREAM_CHUNK_SIZE):
                    logger.info(f"Raw AI service chunk: {chunk!r}")
                    buffer.extend(chunk)
                    
                    # Process complete lines
                    while b'\n' in buffer:
                        line_bytes, buffer = buffer.split(b'\n', 1)
                        line = line_bytes.decode('utf-8')
                        if line.strip():
                            yield f"{line}\n\n"
                
                # Process any remaining data in the buffer
                if buffer:
                    remaining = buffer.decode('utf-8').strip()
                    if remaining:
                        yield f"{remaining}\n\n"
                    
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            yield f"data: {{\"error\":{{\"message\":\"{str(e)}\",\"code\":500}}}}\n\n"

# Request Processor
class RequestProcessor:
    """Process requests sequentially using a queue to accommodate single-threaded backends."""
    
    queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
    processing_lock = asyncio.Lock()
    
    # Define which endpoints need to be processed sequentially
    MODEL_ENDPOINTS = {
        "/v1/chat/completions": (ChatCompletionRequest, ServiceHandler.generate_text_response),
        "/v1/embeddings": (EmbeddingRequest, ServiceHandler.generate_embeddings_response),
        "/chat/completions": (ChatCompletionRequest, ServiceHandler.generate_text_response),
        "/embeddings": (EmbeddingRequest, ServiceHandler.generate_embeddings_response),
    }
    
    @staticmethod
    async def _ensure_server_running(request_id: str, requested_model_id: Optional[str] = None) -> bool:
        """Ensure the AI server for this instance's model is running, reload if necessary.
        The requested_model_id is used for assertion.
        """
        this_instance_service_info = get_this_instance_model_service_info() # Get current instance's info (raises 503 if not set)
        this_instance_model_name = this_instance_service_info.get("model_name")

        if not this_instance_model_name: # Should be caught by getter, but defensive
            logger.error(f"[{request_id}] This instance's model name is not set. Cannot ensure server status.")
            return False

        if requested_model_id and requested_model_id != this_instance_model_name:
            logger.error(f"[{request_id}] _ensure_server_running called for model '{requested_model_id}' but this instance serves '{this_instance_model_name}'. This indicates a routing or logic error.")
            # This is a critical mismatch. This instance cannot ensure server for a different model.
            raise HTTPException(status_code=500, detail=f"Server instance routing mismatch: cannot ensure status for '{requested_model_id}'.")

        target_model_name_for_log = this_instance_model_name
        logger.debug(f"[{request_id}] Ensuring server for model '{target_model_name_for_log}' (this instance) is running.")

        try:
            current_pid = this_instance_service_info.get("pid")
            if not current_pid or not psutil.pid_exists(current_pid):
                logger.info(f"[{request_id}] AI server for model '{target_model_name_for_log}' (PID: {current_pid}) not running or PID not found, reloading...")
                # reload_ai_server will act on this instance's model. Pass name for assertion.
                return await ServiceHandler.reload_ai_server(model_id=this_instance_model_name)
            return True
        except HTTPException as e:
            # This might happen if get_this_instance_model_service_info itself fails, though it's called above.
            # Or if reload_ai_server calls it again and it fails.
            logger.warning(f"[{request_id}] HTTPException while ensuring server for '{target_model_name_for_log}': {e.detail}. Cannot ensure server is running.")
            return False
    
    @staticmethod
    async def process_request(endpoint: str, request_data: Dict[str, Any], model_id_for_service: Optional[str]):
        """Process a request by adding it to the queue and waiting for the result."""
        request_id = generate_request_id() # For logging
        
        if not model_id_for_service:
            # This case should ideally be handled by the calling endpoint or schema default.
            logger.error(f"[{request_id}] Critical: model_id_for_service is None for endpoint {endpoint}. Request cannot be processed without a target model.")
            raise HTTPException(status_code=400, detail="Target model ID not specified in the request.")

        logger.info(f"[{request_id}] Adding request to queue for endpoint {endpoint}, model '{model_id_for_service}' (queue size: {RequestProcessor.queue.qsize()})")
        
        # Update last_activity for the model served by this instance.
        # model_id_for_service should match this instance's model name.
        if hasattr(app.state, "service_info") and app.state.service_info :
            this_instance_model_name = app.state.service_info.get("model_name")
            if this_instance_model_name == model_id_for_service:
                app.state.service_info["last_activity"] = time.time()
                logger.debug(f"Updated last_activity for model '{model_id_for_service}' (this instance).")
            else:
                logger.warning(f"[{request_id}] Attempted to update last_activity for '{model_id_for_service}', but instance serves '{this_instance_model_name}'. Global last_request_time updated instead.")
                app.state.last_request_time = time.time() # Fallback to global if mismatch
        else:
            # This case means service_info isn't set, which get_this_instance_model_service_info should prevent.
            logger.warning(f"[{request_id}] service_info not set when trying to update last_activity for '{model_id_for_service}'. Global last_request_time updated.")
            app.state.last_request_time = time.time() # Fallback

        # Ensure the specific model's server is running
        await RequestProcessor._ensure_server_running(request_id, model_id_for_service)
        
        start_wait_time = time.time()
        future = asyncio.Future()
        # Pass model_id_for_service to the worker via the queue
        await RequestProcessor.queue.put((endpoint, request_data, future, request_id, start_wait_time, model_id_for_service))
        
        logger.info(f"[{request_id}] Waiting for result from endpoint {endpoint}")
        result = await future
        
        total_time = time.time() - start_wait_time
        logger.info(f"[{request_id}] Request completed for endpoint {endpoint} (total time: {total_time:.2f}s)")
        
        return result
    
    @staticmethod
    async def process_direct(endpoint: str, request_data: Dict[str, Any], model_id_for_service: Optional[str]):
        """Process a request directly without queueing for administrative endpoints."""
        request_id = generate_request_id()
        
        if not model_id_for_service:
            logger.error(f"[{request_id}] Critical: model_id_for_service is None for direct processing of {endpoint}.")
            raise HTTPException(status_code=400, detail="Target model ID not specified for direct processing.")

        logger.info(f"[{request_id}] Processing direct request for endpoint {endpoint}, model '{model_id_for_service}'")

        # Update last_activity for the model served by this instance.
        if hasattr(app.state, "service_info") and app.state.service_info:
            this_instance_model_name = app.state.service_info.get("model_name")
            if this_instance_model_name == model_id_for_service:
                app.state.service_info["last_activity"] = time.time()
            else: # Mismatch
                logger.warning(f"[{request_id}] process_direct: Mismatch. Instance serves '{this_instance_model_name}', request for '{model_id_for_service}'. Global last_request_time updated.")
                app.state.last_request_time = time.time()
        else: # service_info not set
            logger.warning(f"[{request_id}] process_direct: service_info not set. Global last_request_time updated.")
            app.state.last_request_time = time.time()

        await RequestProcessor._ensure_server_running(request_id, model_id_for_service)
        
        start_time = time.time()
        if endpoint in RequestProcessor.MODEL_ENDPOINTS:
            model_cls, handler_func = RequestProcessor.MODEL_ENDPOINTS[endpoint]
            request_obj = model_cls(**request_data)
            # The handler_func (e.g., ServiceHandler.generate_text_response) now needs model_id.
            # model_id_for_service is already available.
            result = await handler_func(request_obj, model_id_for_service) # Pass model_id to handler
            
            process_time = time.time() - start_time
            logger.info(f"[{request_id}] Direct request completed for endpoint {endpoint}, model '{model_id_for_service}' (time: {process_time:.2f}s)")
            
            return result
        else:
            logger.error(f"[{request_id}] Endpoint not found: {endpoint}")
            raise HTTPException(status_code=404, detail="Endpoint not found")
    
    @staticmethod
    async def worker():
        """Worker function to process requests from the queue sequentially."""
        logger.info("Request processor worker started")
        processed_count = 0
        
        while True:
            try:
                # Added model_id_for_service to the queue item structure
                endpoint, request_data, future, request_id, start_wait_time, model_id_for_service = await RequestProcessor.queue.get()
                
                wait_time = time.time() - start_wait_time
                queue_size = RequestProcessor.queue.qsize()
                processed_count += 1
                
                logger.info(f"[{request_id}] Processing request from queue for endpoint {endpoint}, model '{model_id_for_service}' "
                           f"(wait time: {wait_time:.2f}s, queue size: {queue_size}, processed: {processed_count})")
                
                # The processing_lock is global. If different models can be processed truly in parallel by their
                # llama-server instances, a per-model lock or a more sophisticated concurrency control might be needed.
                # For now, sticking to existing global lock, meaning one request processed at a time by this API layer.
                async with RequestProcessor.processing_lock:
                    processing_start = time.time()
                    
                    if endpoint in RequestProcessor.MODEL_ENDPOINTS:
                        model_cls, handler_func = RequestProcessor.MODEL_ENDPOINTS[endpoint]
                        try:
                            request_obj = model_cls(**request_data)

                            if not model_id_for_service: # Should be passed from process_request
                                # Fallback: try to get from request_obj if somehow missed (should not happen)
                                model_id_for_service = getattr(request_obj, 'model', None)
                                logger.warning(f"[{request_id}] model_id_for_service was None in worker, trying to get from request_obj: {model_id_for_service}")

                            if not model_id_for_service:
                                logger.error(f"[{request_id}] Critical: model_id_for_service is None for queued processing of {endpoint}. Cannot select model.")
                                raise HTTPException(status_code=400, detail="Model ID not specified for queued processing.")

                            result = await handler_func(request_obj, model_id_for_service) # Pass model_id to handler
                            future.set_result(result)
                            
                            processing_time = time.time() - processing_start
                            total_time = time.time() - start_wait_time
                            
                            logger.info(f"[{request_id}] Completed request for endpoint {endpoint}, model '{model_id_for_service}' "
                                       f"(processing: {processing_time:.2f}s, total: {total_time:.2f}s)")
                        except Exception as e:
                            logger.error(f"[{request_id}] Handler error for {endpoint}, model '{model_id_for_service}': {str(e)}")
                            future.set_exception(e)
                    else:
                        logger.error(f"[{request_id}] Endpoint not found: {endpoint}")
                        future.set_exception(HTTPException(status_code=404, detail="Endpoint not found"))
                
                RequestProcessor.queue.task_done()
                
                # Log periodic status about queue health
                if processed_count % 10 == 0:
                    logger.info(f"Queue status: current size={queue_size}, processed={processed_count}")
                
            except asyncio.CancelledError:
                logger.info("Worker task cancelled, exiting")
                break
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")

# Background Tasks
async def unload_checker():
    """Periodically check if the AI server has been idle for too long and unload it if needed."""
    logger.info("Unload checker task started")
    
    while True:
        try:
            await asyncio.sleep(UNLOAD_CHECK_INTERVAL)

            # Unload checker for a single FastAPI instance per model
            if not hasattr(app.state, "service_info") or not app.state.service_info : # service_info not set for this instance
                logger.debug("Unload checker: No service_info for this instance.")
                continue

            current_time = time.time()
            # app.state.service_info is this instance's model metadata
            this_model_details = get_this_instance_model_service_info() # Use getter for validation
            model_name = this_model_details.get("model_name")

            if not model_name:
                logger.warning("Unload checker: This instance's model name is not set in service_info. Cannot proceed with unload check.")
                continue

            # Use 'last_activity' from this instance's service_info.
            last_activity = this_model_details.get("last_activity", current_time) # Default to current_time if no activity logged
            pid = this_model_details.get("pid")

            if not pid or not psutil.pid_exists(pid):
                if pid:
                    logger.info(f"Unload checker: Model '{model_name}' (this instance, PID: {pid}) process not found. Clearing PID from instance state.")
                    app.state.service_info.pop("pid", None)
                else:
                    logger.debug(f"Unload checker: Model '{model_name}' (this instance) has no PID or process is not running. Nothing to unload.")
                continue

            idle_time = current_time - last_activity
            logger.debug(f"Unload check for model '{model_name}' (this instance): idle_time={idle_time:.2f}s, PID={pid}")

            if idle_time > IDLE_TIMEOUT:
                logger.info(f"Model '{model_name}' (this instance, PID: {pid}) has been idle for {idle_time:.2f}s. Unloading.")
                try:
                    # kill_ai_server will use this instance's model_name if model_id is None,
                    # or can assert if model_id (model_name) is passed.
                    killed = await ServiceHandler.kill_ai_server(model_id=model_name)
                    if killed:
                        logger.info(f"Successfully unloaded idle model '{model_name}' for this instance.")
                    else:
                        logger.warning(f"Failed to unload idle model '{model_name}' (this instance) or already stopped.")
                except Exception as e:
                    logger.error(f"Error while trying to unload model '{model_name}' (this instance): {getattr(e, 'detail', e)}", exc_info=True)

        except asyncio.CancelledError:
            logger.info("Unload checker task cancelled")
            
        except asyncio.CancelledError:
            logger.info("Unload checker task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in unload checker task: {str(e)}", exc_info=True)

# Performance monitoring middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware that adds a header with the processing time for the request."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Lifecycle Events
@app.on_event("startup")
async def startup_event():
    """Startup event handler: initialize the HTTP client and start the worker task."""
    # Create an asynchronous HTTP client with optimized connection pooling
    limits = httpx.Limits(
        max_connections=POOL_CONNECTIONS,
        max_keepalive_connections=POOL_CONNECTIONS,
        keepalive_expiry=POOL_KEEPALIVE
    )
    app.state.client = httpx.AsyncClient(
        limits=limits,
        timeout=HTTP_TIMEOUT,
        transport=httpx.AsyncHTTPTransport(
            retries=MAX_RETRIES,
            verify=False  # Disable SSL verification for local connections
        )
    )
    
    # Initialize the last request time
    app.state.last_request_time = time.time()
    app.state.service_info = {} # Ensure service_info is initialized as a dict
    
    # Start background tasks
    app.state.worker_task = asyncio.create_task(RequestProcessor.worker())
    app.state.unload_checker_task = asyncio.create_task(unload_checker())
    
    logger.info("Service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the application shuts down."""
    # Cancel background tasks
    tasks_to_cancel = []
    for task_attr in ["worker_task", "unload_checker_task"]:
        if hasattr(app.state, task_attr):
            task = getattr(app.state, task_attr)
            task.cancel()
            tasks_to_cancel.append(task)
    
    # Wait for tasks to complete cancellation
    if tasks_to_cancel:
        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
    
    # Shutdown event for a single FastAPI instance per model
    logger.info("Shutting down: stopping AI model service for this instance.")
    if hasattr(app.state, "service_info") and app.state.service_info:
        try:
            # service_info is this instance's model details
            model_details = get_this_instance_model_service_info() # Use getter for validation
            model_name = model_details.get("model_name")

            if model_name:
                logger.info(f"Attempting to stop server for model '{model_name}' (this instance) during shutdown.")
                # Pass model_name for assertion by kill_ai_server
                await ServiceHandler.kill_ai_server(model_id=model_name)
            else: # Should be caught by getter, but defensive.
                logger.warning("No model_name found in service_info for this instance at shutdown.")
        except HTTPException as e: # e.g. if service_info was not set or malformed
             logger.error(f"Could not get model details for shutdown: {e.detail}")
        except Exception as e:
            logger.error(f"Unexpected error during shutdown for instance's model: {e}", exc_info=True)
    else:
        logger.info("No service info found for this instance, no specific model to stop.")
    
    # Close HTTP client
    if hasattr(app.state, "client"):
        await app.state.client.aclose()
    
    logger.info("Service shutdown complete")

# API Endpoints
@app.get("/health")
@app.get("/v1/health")
async def health():
    """Health check endpoint that bypasses the request queue for immediate response."""
    return {"status": "ok"}

@app.post("/update")
async def update(model_specific_metadata: Dict[str, Any]):
    """
    Update/set the service information for the single model this FastAPI instance serves.
    Receives metadata for that one model from LocalAIManager.
    """
    if not isinstance(model_specific_metadata, dict):
        raise HTTPException(status_code=400, detail="Invalid service info format; expected a dictionary for a single model's metadata.")

    # model_hash = model_specific_metadata.get("hash") # Not used as key if only one model's info is stored
    model_name = model_specific_metadata.get("model_name")

    if not model_name: # Also hash, port etc. are essential, but model_name is key for user.
        raise HTTPException(status_code=400, detail="Received service info must include at least 'model_name'.")

    # This FastAPI instance serves one model, so its app.state.service_info is this model's metadata.
    app.state.service_info = model_specific_metadata
    logger.info(f"Service info updated for model '{model_name}'. This instance now serves this model and its state is set.")
    return {"status": "ok", "message": "Service info updated successfully"}

@app.get("/v1/models")
async def list_models():
    """
    List the model served by this FastAPI instance.
    In a one-FastAPI-per-model architecture, this endpoint lists only the
    single model this instance is configured for.
    """
    # get_this_instance_model_service_info() will raise 503 if not configured.
    # If it returns, then service_info is valid.
    try:
        model_details = get_this_instance_model_service_info()
    except HTTPException as e:
        if e.status_code == 503: # Not yet configured by LocalAIManager
             return {"object": "list", "data": []}
        raise # Re-raise other HttpExceptions (e.g. malformed)

    model_id = model_details.get("model_name")

    if not model_id:
        # This should ideally be caught by get_this_instance_model_service_info's validation,
        # but as an extra safeguard for the list structure.
        logger.error("This instance's model is missing 'model_name' in its service_info for /v1/models after validation.")
        return {"object": "list", "data": []} # Should not happen if getter works

    # 'created_time' was a placeholder. 'last_activity' or a dedicated 'load_time' would be better.
    # LocalAIManager sends 'last_activity' which is updated dynamically.
    # For "created" time of the model card, a fixed timestamp from metadata or load time is more suitable.
    # Using 'last_activity' here might be confusing as "created" time.
    # Let's use a fixed value or a value from service_info if LocalAI manager adds e.g. "model_load_timestamp"
    created_timestamp = int(model_details.get("model_load_timestamp", model_details.get("last_activity", time.time())))

    models_data = [{
        "id": model_id,
        "object": "model",
        "created": created_timestamp,
        "owned_by": "local-ai", # Placeholder
    }]

    return {"object": "list", "data": models_data}

# Model-based endpoints that use the request queue
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Endpoint for chat completion requests."""
    # request.model should be Optional[str]. If None, a default should be chosen.
    # The schema change made it Optional[str] = None.
    # The API layer or RequestProcessor should handle "None" to mean a default model.
    model_id_from_req = request.model
    if model_id_from_req is None:
        # TODO: Determine default model logic. For now, raise error or use a hardcoded default.
        # This could involve checking if it's a vision request to pick vision default, etc.
        # For now, let's assume a text default if not specified.
        logger.warning("Model ID not specified in /chat/completions request, using default text model.")
        # model_id_from_req = Config.TEXT_MODEL # Assuming Config is available or use string
        # For now, let's ensure it's explicitly provided until default logic is clear for multi-model.
        raise HTTPException(status_code=400, detail="Model ID must be specified in the request for multi-model setup.")

    request_dict = convert_request_to_dict(request)
    return await RequestProcessor.process_request("/chat/completions", request_dict, model_id_from_req)

@app.post("/embeddings")
async def embeddings(request: EmbeddingRequest):
    """Endpoint for embedding requests."""
    model_id_from_req = request.model
    if model_id_from_req is None:
        logger.warning("Model ID not specified in /embeddings request, using default embedding model.")
        # model_id_from_req = Config.EMBEDDING_MODEL # Assuming Config is available
        raise HTTPException(status_code=400, detail="Model ID must be specified in the request for multi-model setup.")

    request_dict = convert_request_to_dict(request)
    return await RequestProcessor.process_request("/embeddings", request_dict, model_id_from_req)

@app.post("/v1/chat/completions")
async def v1_chat_completions(request: ChatCompletionRequest):
    """Endpoint for chat completion requests (v1 API)."""
    model_id_from_req = request.model
    if model_id_from_req is None:
        # Default model logic: if vision, use vision model, else text model.
        # This requires access to Config and request.is_vision_request().
        # from local_ai.schema import Config # Ensure Config is imported if not already
        # if request.is_vision_request():
        #     model_id_from_req = Config.VISION_MODEL
        # else:
        #     model_id_from_req = Config.TEXT_MODEL
        # logger.info(f"Model ID not specified, defaulted to: {model_id_from_req}")
        # For now, require explicit model ID.
        raise HTTPException(status_code=400, detail="Model ID must be specified in the request for /v1/chat/completions.")

    request_dict = convert_request_to_dict(request)
    return await RequestProcessor.process_request("/v1/chat/completions", request_dict, model_id_from_req)

@app.post("/v1/embeddings")
async def v1_embeddings(request: EmbeddingRequest):
    """Endpoint for embedding requests (v1 API)."""
    model_id_from_req = request.model
    if model_id_from_req is None:
        # from local_ai.schema import Config # Ensure Config is imported
        # model_id_from_req = Config.EMBEDDING_MODEL
        # logger.info(f"Model ID not specified, defaulted to: {model_id_from_req}")
        raise HTTPException(status_code=400, detail="Model ID must be specified in the request for /v1/embeddings.")

    request_dict = convert_request_to_dict(request)
    return await RequestProcessor.process_request("/v1/embeddings", request_dict, model_id_from_req)