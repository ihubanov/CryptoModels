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
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from functools import lru_cache

# Import schemas from schema.py
from local_ai.schema import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse
)

# Set up logging with both console and file output
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI()

# Constants for dynamic unload feature
IDLE_TIMEOUT = 600  # 10 minutes in seconds
UNLOAD_CHECK_INTERVAL = 60  # Check every 60 seconds
SERVICE_START_TIMEOUT = 60  # Maximum time to wait for service to start
POOL_CONNECTIONS = 1000  # Increased from 100 to handle more concurrent connections
POOL_KEEPALIVE = 60  # Increased from 20 to reduce connection churn
HTTP_TIMEOUT = 300.0  # Default timeout for HTTP requests in seconds
STREAM_TIMEOUT = 300.0  # Default timeout for streaming requests in seconds
MAX_RETRIES = 3  # Maximum number of retries for failed requests
RETRY_DELAY = 1.0  # Delay between retries in seconds

# Cache for service port to avoid repeated lookups
@lru_cache(maxsize=1)
def get_cached_service_port():
    """
    Retrieve the port of the underlying service from the app's state with caching.
    The cache is invalidated when the service info is updated.
    """
    if not hasattr(app.state, "service_info") or "port" not in app.state.service_info:
        logger.error("Service information not set")
        raise HTTPException(status_code=503, detail="Service information not set")
    return app.state.service_info["port"]

# Add request caching
@lru_cache(maxsize=1000)
def cache_request(request_data: str) -> str:
    """
    Cache request data to avoid processing duplicate requests.
    """
    return request_data

# Service Functions
class ServiceHandler:
    """
    Handler class for making requests to the underlying service.
    """
    @staticmethod
    async def get_service_port() -> int:
        """
        Retrieve the port of the underlying service from the app's state.
        """
        try:
            return get_cached_service_port()
        except HTTPException:
            # If cache lookup fails, try direct lookup
            if not hasattr(app.state, "service_info") or "port" not in app.state.service_info:
                logger.error("Service information not set")
                raise HTTPException(status_code=503, detail="Service information not set")
            return app.state.service_info["port"]
    
    @staticmethod
    async def kill_ai_server():
        """
        Kill the AI server process if it's running.
        """
        try:
            # Get the PID from the service info
            if not hasattr(app.state, "service_info") or "pid" not in app.state.service_info:
                logger.warning("No PID found in service info, cannot kill AI server")
                return False
                
            pid = app.state.service_info["pid"]
            logger.info(f"Attempting to kill AI server with PID {pid}")
            
            # Try to kill the process group (more reliable than just the process)
            try:
                # First try to get the process group ID
                pgid = os.getpgid(pid)
                # Kill the entire process group
                os.killpg(pgid, signal.SIGTERM)
                logger.info(f"Successfully sent SIGTERM to process group {pgid}")
            except (ProcessLookupError, OSError) as e:
                logger.warning(f"Could not kill process group: {e}")
                # Fall back to killing just the process
                os.kill(pid, signal.SIGTERM)
                logger.info(f"Successfully sent SIGTERM to process {pid}")
            
            # Wait a moment and check if the process is still running
            await asyncio.sleep(2)
            try:
                os.kill(pid, 0)  # Check if process exists
                # If we get here, the process is still running, try SIGKILL
                logger.warning(f"Process {pid} still running after SIGTERM, sending SIGKILL")
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                # Process is already gone, which is good
                pass
                
            # Remove the PID from service info
            if hasattr(app.state, "service_info"):
                app.state.service_info.pop("pid", None)
                
            return True
        except Exception as e:
            logger.error(f"Error killing AI server: {str(e)}", exc_info=True)
            return False
    
    @staticmethod
    async def reload_ai_server(service_start_timeout: int = SERVICE_START_TIMEOUT):
        """
        Reload the AI server process.
        
        Args:
            service_start_timeout: Maximum time in seconds to wait for the service to start
            
        Returns:
            bool: True if the service was successfully started, False otherwise
        """
        try:
            # Get the command to start AI server from the service info
            if not hasattr(app.state, "service_info") or "running_ai_command" not in app.state.service_info:
                logger.error("No running_ai_command found in service info, cannot reload AI server")
                return False
                
            running_ai_command = app.state.service_info["running_ai_command"]
            logger.info(f"Reloading AI server with command: {running_ai_command}")

            logs_dir = Path("logs")
            # Ensure logs directory exists
            logs_dir.mkdir(exist_ok=True)
            
            # Set up log file
            ai_log_stderr = logs_dir / "ai.log"
            ai_process = None
            
            try:
                with open(ai_log_stderr, 'w') as stderr_log:
                    ai_process = subprocess.Popen(
                        running_ai_command,
                        stderr=stderr_log,
                        preexec_fn=os.setsid  # Run in a new process group
                    )
                logger.info(f"AI logs written to {ai_log_stderr}")
            except Exception as e:
                logger.error(f"Error starting AI service: {str(e)}", exc_info=True)
                return False
            
            # Wait for the process to start by checking the health endpoint
            port = app.state.service_info["port"]
            start_time = time.time()
            health_check_interval = 1  # Check every second
            
            while time.time() - start_time < service_start_timeout:
                try:
                    response = requests.get(f"http://localhost:{port}/health", timeout=2)
                    if response.status_code == 200:
                        logger.info(f"Service health check passed after {time.time() - start_time:.2f}s")
                        break
                except (requests.RequestException, ConnectionError) as e:
                    # Just wait and try again
                    await asyncio.sleep(health_check_interval)
            
            # Check if the process is running
            if ai_process.poll() is None:
                # Process is running, update the PID in service info
                if hasattr(app.state, "service_info"):
                    app.state.service_info["pid"] = ai_process.pid
                logger.info(f"Successfully reloaded AI server with PID {ai_process.pid}")
                return True
            else:
                # Process failed to start
                logger.error(f"Failed to reload AI server: Process exited with code {ai_process.returncode}")
                return False
        except Exception as e:
            logger.error(f"Error reloading AI server: {str(e)}", exc_info=True)
            return False
    
    @staticmethod
    async def generate_text_response(request: ChatCompletionRequest):
        """
        Generate a response for chat completion requests, supporting both streaming and non-streaming.
        """
        port = await ServiceHandler.get_service_port()
        if request.is_vision_request():
            if not app.state.service_info["multimodal"]:
                raise HTTPException(status_code=400, detail="Vision-based requests are not supported for this model")
            request.fix_messages()
        # Convert to dict, supporting both Pydantic v1 and v2
        request_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        
        if request.stream:
            # Return a streaming response for non-tool requests
            return StreamingResponse(
                ServiceHandler._stream_generator(port, request_dict),
                media_type="text/event-stream"
            )

        # Make a non-streaming API call
        response_data = await ServiceHandler._make_api_call(port, "/v1/chat/completions", request_dict)
        assert isinstance(response_data, dict), "Response data must be a dictionary"
        return ChatCompletionResponse(
            id=response_data.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
            object=response_data.get("object", "chat.completion"),
            created=response_data.get("created", int(time.time())),
            model=request.model,
            choices=response_data.get("choices", [])
        )
    
    @staticmethod
    async def generate_embeddings_response(request: EmbeddingRequest):
        """
        Generate a response for embedding requests.
        """
        port = await ServiceHandler.get_service_port()
        # Convert to dict, supporting both Pydantic v1 and v2
        request_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        response_data = await ServiceHandler._make_api_call(port, "/v1/embeddings", request_dict)
        assert isinstance(response_data, dict), "Response data must be a dictionary"
        return EmbeddingResponse(
            object=response_data.get("object", "list"),
            data=response_data.get("data", []),
            model=request.model
        )
    
    @staticmethod
    async def _make_api_call(port: int, endpoint: str, data: dict) -> dict:
        """
        Make a non-streaming API call to the specified endpoint and return the JSON response.
        Includes retry logic for transient errors.
        """
        
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
                # Don't retry client errors (4xx), only server errors (5xx)
                if response.status_code < 500:
                    raise HTTPException(status_code=response.status_code, detail=error_text)
            else:
                return response.json()
        except httpx.TimeoutException as e:
            raise HTTPException(status_code=504, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    
    @staticmethod
    async def _stream_generator(port: int, data: dict):
        """
        Generator for streaming responses from the service.
        Yields chunks of data as they are received, formatted for SSE (Server-Sent Events).
        """
        try:
            # Use a larger chunk size for better performance
            async with app.state.client.stream(
                "POST", 
                f"http://localhost:{port}/v1/chat/completions", 
                json=data,
                timeout=STREAM_TIMEOUT
            ) as response:
                if response.status_code != 200:
                    error_text = await response.text()
                    error_msg = f"data: {{\"error\":{{\"message\":\"{error_text}\",\"code\":{response.status_code}}}}}\n\n"
                    logger.error(f"Streaming error: {response.status_code} - {error_text}")
                    yield error_msg
                    return
                
                buffer = ""
                async for chunk in response.aiter_bytes():
                    buffer += chunk.decode('utf-8')
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if line.strip():
                            yield f"{line}\n\n"
                
                # Process any remaining data in the buffer
                if buffer.strip():
                    yield f"{buffer}\n\n"
                    
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            yield f"data: {{\"error\":{{\"message\":\"{str(e)}\",\"code\":500}}}}\n\n"

    @staticmethod
    async def _fake_stream_with_tools(formatted_response: dict, model: str):
        """
        Generate a fake streaming response for tool-based chat completions.
        This method simulates the streaming behavior by breaking a complete response into chunks.
        
        Args:
            formatted_response: The complete response to stream in chunks
            model: The model name
        """
        # Pre-compute common values
        base_chunk = {
            "id": formatted_response.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
            "object": "chat.completion.chunk",
            "created": formatted_response.get("created", int(time.time())),
            "model": formatted_response.get("model", model),
        }
        
        if "system_fingerprint" in formatted_response:
            base_chunk["system_fingerprint"] = formatted_response["system_fingerprint"]

        choices = formatted_response.get("choices", [])
        if not choices:
            yield f"data: {json.dumps({**base_chunk, 'choices': []})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Pre-compute all chunks
        chunks = []
        
        # Initial chunk with role
        initial_choices = [
            {
                "index": choice["index"],
                "delta": {"role": "assistant", "content": ""},
                "logprobs": None,
                "finish_reason": None
            }
            for choice in choices
        ]
        chunks.append({**base_chunk, 'choices': initial_choices})

        # Content chunk
        content_choices = []
        for choice in choices:
            message = choice.get("message", {})
            delta = {}
            
            if "tool_calls" in message:
                updated_tool_calls = [
                    {**tool_call, "index": str(idx)}
                    for idx, tool_call in enumerate(message["tool_calls"])
                ]
                delta["tool_calls"] = updated_tool_calls
            elif message.get("content"):
                delta["content"] = message.get("content", "")
            else:
                delta["content"] = ""
                
            if delta:
                content_choices.append({
                    "index": choice["index"],
                    "delta": delta,
                    "logprobs": None,
                    "finish_reason": None
                })
                
        if content_choices:
            chunks.append({**base_chunk, 'choices': content_choices})

        # Final chunk with finish reason
        finish_choices = [
            {
                "index": choice["index"],
                "delta": {},
                "logprobs": None,
                "finish_reason": choice["finish_reason"]
            }
            for choice in choices
        ]
        chunks.append({**base_chunk, 'choices': finish_choices})

        # Yield all chunks
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

# Request Processor
class RequestProcessor:
    """
    Class for processing requests sequentially using a queue.
    Ensures that only one request is processed at a time to accommodate limitations
    of backends like llama-server that can only handle one request at a time.
    """
    queue = asyncio.Queue(maxsize=1000)  # Added maxsize to prevent memory issues
    processing_lock = asyncio.Lock()  # Lock to ensure only one request is processed at a time
    
    # Define which endpoints need to be processed sequentially
    MODEL_ENDPOINTS = {
        "/v1/chat/completions": (ChatCompletionRequest, ServiceHandler.generate_text_response),
        "/v1/embeddings": (EmbeddingRequest, ServiceHandler.generate_embeddings_response),
        "/chat/completions": (ChatCompletionRequest, ServiceHandler.generate_text_response),
        "/embeddings": (EmbeddingRequest, ServiceHandler.generate_embeddings_response),
    }  # Mapping of endpoints to their request models and handlers
    
    @staticmethod
    async def process_request(endpoint: str, request_data: dict):
        """
        Process a request by adding it to the queue and waiting for the result.
        This ensures requests are processed in order, one at a time.
        Returns a Future that will be resolved with the result.
        """
        request_id = str(uuid.uuid4())[:8]  # Generate a short request ID for tracking
        queue_size = RequestProcessor.queue.qsize()
        
        logger.info(f"[{request_id}] Adding request to queue for endpoint {endpoint} (queue size: {queue_size})")
        
        # Update the last request time
        app.state.last_request_time = time.time()
        
        # Check if we need to reload the AI server
        if hasattr(app.state, "service_info") and "pid" not in app.state.service_info:
            logger.info(f"[{request_id}] AI server not running, reloading...")
            await ServiceHandler.reload_ai_server()
        
        start_wait_time = time.time()
        future = asyncio.Future()
        await RequestProcessor.queue.put((endpoint, request_data, future, request_id, start_wait_time))
        
        # Wait for the future to be resolved
        logger.info(f"[{request_id}] Waiting for result from endpoint {endpoint}")
        result = await future
        
        total_time = time.time() - start_wait_time
        logger.info(f"[{request_id}] Request completed for endpoint {endpoint} (total time: {total_time:.2f}s)")
        
        return result
    
    @staticmethod
    async def process_direct(endpoint: str, request_data: dict):
        """
        Process a request directly without queueing.
        Use this for administrative endpoints that don't require model access.
        """
        request_id = str(uuid.uuid4())[:8]  # Generate a short request ID for tracking
        logger.info(f"[{request_id}] Processing direct request for endpoint {endpoint}")
        
        # Update the last request time
        app.state.last_request_time = time.time()
        
        # Check if we need to reload the AI server
        if hasattr(app.state, "service_info") and "pid" not in app.state.service_info:
            logger.info(f"[{request_id}] AI server not running, reloading...")
            await ServiceHandler.reload_ai_server()
        
        start_time = time.time()
        if endpoint in RequestProcessor.MODEL_ENDPOINTS:
            model_cls, handler = RequestProcessor.MODEL_ENDPOINTS[endpoint]
            request_obj = model_cls(**request_data)
            result = await handler(request_obj)
            
            process_time = time.time() - start_time
            logger.info(f"[{request_id}] Direct request completed for endpoint {endpoint} (time: {process_time:.2f}s)")
            
            return result
        else:
            logger.error(f"[{request_id}] Endpoint not found: {endpoint}")
            raise HTTPException(status_code=404, detail="Endpoint not found")
    
    # Global worker function
    @staticmethod
    async def worker():
        """
        Worker function to process requests from the queue sequentially.
        Only one request is processed at a time.
        """
        logger.info("Request processor worker started")
        processed_count = 0
        
        while True:
            try:
                endpoint, request_data, future, request_id, start_wait_time = await RequestProcessor.queue.get()
                
                wait_time = time.time() - start_wait_time
                queue_size = RequestProcessor.queue.qsize()
                processed_count += 1
                
                logger.info(f"[{request_id}] Processing request from queue for endpoint {endpoint} "
                           f"(wait time: {wait_time:.2f}s, queue size: {queue_size}, processed: {processed_count})")
                
                # Use the lock to ensure only one request is processed at a time
                async with RequestProcessor.processing_lock:
                    processing_start = time.time()
                    
                    if endpoint in RequestProcessor.MODEL_ENDPOINTS:
                        model_cls, handler = RequestProcessor.MODEL_ENDPOINTS[endpoint]
                        try:
                            request_obj = model_cls(**request_data)
                            result = await handler(request_obj)
                            future.set_result(result)
                            
                            processing_time = time.time() - processing_start
                            total_time = time.time() - start_wait_time
                            
                            logger.info(f"[{request_id}] Completed request for endpoint {endpoint} "
                                       f"(processing: {processing_time:.2f}s, total: {total_time:.2f}s)")
                        except Exception as e:
                            logger.error(f"[{request_id}] Handler error for {endpoint}: {str(e)}")
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
                break  # Exit the loop when the task is canceled
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")
                # Continue working, don't crash the worker

# Unload checker task
async def unload_checker():
    """
    Periodically check if the AI server has been idle for too long and unload it if needed.
    """
    logger.info("Unload checker task started")
    
    while True:
        try:
            # Wait for the check interval
            await asyncio.sleep(UNLOAD_CHECK_INTERVAL)   
            # Check if the service is running and has been idle for too long
            if (hasattr(app.state, "service_info") and 
                "pid" in app.state.service_info and 
                hasattr(app.state, "last_request_time")):
                
                idle_time = time.time() - app.state.last_request_time
                
                if idle_time > IDLE_TIMEOUT:
                    logger.info(f"AI server has been idle for {idle_time:.2f}s, unloading...")
                    await ServiceHandler.kill_ai_server()
            
        except asyncio.CancelledError:
            logger.info("Unload checker task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in unload checker task: {str(e)}", exc_info=True)
            # Continue running despite errors

# Performance monitoring middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware that adds a header with the processing time for the request.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Lifecycle Events
@app.on_event("startup")
async def startup_event():
    """
    Startup event handler: initialize the HTTP client and start the worker task.
    """
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
    
    # Start the worker
    app.state.worker_task = asyncio.create_task(RequestProcessor.worker())
    
    # Start the unload checker task
    app.state.unload_checker_task = asyncio.create_task(unload_checker())
    
    logger.info("Service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Clean up resources when the application shuts down.
    """
    # Cancel background tasks
    if hasattr(app.state, "worker_task"):
        app.state.worker_task.cancel()
    if hasattr(app.state, "unload_checker_task"):
        app.state.unload_checker_task.cancel()
    
    # Kill the AI server if it's running
    if hasattr(app.state, "service_info") and "pid" in app.state.service_info:
        await ServiceHandler.kill_ai_server()
    
    logger.info("Service shutdown complete")

# API Endpoints
@app.get("/health")
@app.get("/v1/health")
async def health():
    """
    Health check endpoint.
    Returns a simple status to indicate the service is running.
    This endpoint bypasses the request queue for immediate response.
    """
    return {"status": "ok"}


@app.post("/update")
async def update(request: dict):
    """
    Update the service information in the app's state.
    Stores the provided request data for use in determining the service port.
    This endpoint bypasses the request queue for immediate response.
    """
    app.state.service_info = request
    # Invalidate the cache when service info is updated
    get_cached_service_port.cache_clear()
    return {"status": "ok", "message": "Service info updated successfully"}

# Modified endpoint handlers for model-based endpoints
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Endpoint for chat completion requests.
    Uses the request queue to ensure only one model request is processed at a time.
    """
    # Convert to dict, supporting both Pydantic v1 and v2
    request_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    return await RequestProcessor.process_request("/chat/completions", request_dict)

@app.post("/embeddings")
async def embeddings(request: EmbeddingRequest):
    """
    Endpoint for embedding requests.
    Uses the request queue to ensure only one model request is processed at a time.
    """
    # Convert to dict, supporting both Pydantic v1 and v2
    request_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    return await RequestProcessor.process_request("/embeddings", request_dict)

@app.post("/v1/chat/completions")
async def v1_chat_completions(request: ChatCompletionRequest):
    """
    Endpoint for chat completion requests (v1 API).
    Uses the request queue to ensure only one model request is processed at a time.
    """
    # Convert to dict, supporting both Pydantic v1 and v2
    request_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    return await RequestProcessor.process_request("/v1/chat/completions", request_dict)

@app.post("/v1/embeddings")
async def v1_embeddings(request: EmbeddingRequest):
    """
    Endpoint for embedding requests (v1 API).
    Uses the request queue to ensure only one model request is processed at a time.
    """
    # Convert to dict, supporting both Pydantic v1 and v2
    request_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    return await RequestProcessor.process_request("/v1/embeddings", request_dict)