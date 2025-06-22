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
import psutil
from pathlib import Path
from json_repair import repair_json
from typing import Dict, Any, Optional
from local_ai.utils import wait_for_health
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

# Import schemas from schema.py
from local_ai.schema import (
    Choice,
    Message,
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ChatCompletionChunk,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
    ChatCompletionResponse
)

# Set up logging with both console and file output
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI()

# Constants for dynamic unload feature - Optimized for performance
IDLE_TIMEOUT = 600  # 10 minutes in seconds
UNLOAD_CHECK_INTERVAL = 30  # Check every 30 seconds (reduced from 60)
SERVICE_START_TIMEOUT = 120  # Reduced timeout for faster failure detection
POOL_CONNECTIONS = 50  # Further optimized for better resource usage
POOL_KEEPALIVE = 10  # Reduced keepalive to free connections faster
HTTP_TIMEOUT = 180.0  # Reduced timeout for faster failure detection
STREAM_TIMEOUT = 300.0  # Keep streaming timeout longer for large responses
MAX_RETRIES = 2  # Reduced retries for faster failure handling
RETRY_DELAY = 0.5  # Reduced delay between retries
MAX_QUEUE_SIZE = 50  # Further reduced for better memory usage
HEALTH_CHECK_INTERVAL = 2  # Faster health checks
STREAM_CHUNK_SIZE = 16384  # Increased chunk size for better throughput

# Utility functions
def get_service_info() -> Dict[str, Any]:
    """Get service info from app state with error handling."""
    if not hasattr(app.state, "service_info") or not app.state.service_info:
        raise HTTPException(status_code=503, detail="Service information not set")
    return app.state.service_info

def get_service_port() -> int:
    """Get service port with caching and error handling."""
    service_info = get_service_info()
    if "port" not in service_info:
        raise HTTPException(status_code=503, detail="Service port not configured")
    return service_info["port"]

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
    async def kill_ai_server() -> bool:
        """Kill the AI server process if it's running."""
        try:
            
            service_info = get_service_info()
            if "pid" not in service_info:
                logger.warning("No PID found in service info, cannot kill AI server")
                return False
                
            pid = service_info["pid"]
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
            
            # Clean up service info
            if success:
                service_info.pop("pid", None)
                logger.info("AI server stopped successfully")
                return True
            else:
                logger.error("Failed to stop AI server")
                return False
            
        except Exception as e:
            logger.error(f"Error killing AI server: {str(e)}", exc_info=True)
            return False
    
    @staticmethod
    async def reload_ai_server(service_start_timeout: int = SERVICE_START_TIMEOUT) -> bool:
        """Reload the AI server process."""
        try:
            service_info = get_service_info()
            if "running_ai_command" not in service_info:
                logger.error("No running_ai_command found in service info, cannot reload AI server")
                return False
                
            running_ai_command = service_info["running_ai_command"]
            logger.info(f"Reloading AI server with command: {running_ai_command}")

            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            ai_log_stderr = logs_dir / "ai.log"
            
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
            port = service_info["port"]
            if not wait_for_health(port, timeout=service_start_timeout):
                logger.error(f"AI server failed to start within {service_start_timeout} seconds")
                return False
            
            # Check if the process is running
            if ai_process.poll() is None:
                service_info["pid"] = ai_process.pid
                logger.info(f"Successfully reloaded AI server with PID {ai_process.pid}")
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
    async def generate_text_response(request: ChatCompletionRequest):
        """Generate a response for chat completion requests, supporting both streaming and non-streaming."""
        port = get_service_port()
        
        if request.is_vision_request():
            service_info = get_service_info()
            if not service_info.get("multimodal", False):
                content = "Unfortunately, I'm not equipped to interpret images at this time. Please provide a text description if possible."
                return ServiceHandler._create_vision_error_response(request, content)
                
        request.clean_messages()
        request.enhance_tool_messages()
        request_dict = convert_request_to_dict(request)
        
        if request.stream:
            return StreamingResponse(
                ServiceHandler._stream_generator(port, request_dict),
                media_type="text/event-stream"
            )

        # Make a non-streaming API call
        response_data = await ServiceHandler._make_api_call(port, "/v1/chat/completions", request_dict)
        return ChatCompletionResponse(
            id=response_data.get("id", generate_chat_completion_id()),
            object=response_data.get("object", "chat.completion"),
            created=response_data.get("created", int(time.time())),
            model=request.model,
            choices=response_data.get("choices", [])
        )
    
    @staticmethod
    async def generate_embeddings_response(request: EmbeddingRequest):
        """Generate a response for embedding requests."""
        port = get_service_port()
        request_dict = convert_request_to_dict(request)
        response_data = await ServiceHandler._make_api_call(port, "/v1/embeddings", request_dict)
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
        buffer = ""
        tool_calls = {}
        
        def _extract_json_data(line: str) -> Optional[str]:
            """Extract JSON data from SSE line, return None if not valid data."""
            line = line.strip()
            if not line or line.startswith(': ping'):
                return None
            if line.startswith('data: '):
                return line[6:].strip()
            return None
        
        def _process_tool_call_delta(delta_tool_call, tool_calls: dict):
            """Process tool call delta and update tool_calls dict."""
            tool_call_index = str(delta_tool_call.index)
            if tool_call_index not in tool_calls:
                tool_calls[tool_call_index] = {"arguments": ""}
            
            if delta_tool_call.id is not None:
                tool_calls[tool_call_index]["id"] = delta_tool_call.id
                
            function = delta_tool_call.function
            if function.name is not None:
                tool_calls[tool_call_index]["name"] = function.name
            if function.arguments is not None:
                tool_calls[tool_call_index]["arguments"] += function.arguments
        
        def _create_tool_call_chunks(tool_calls: dict, chunk_obj):
            """Create tool call chunks for final output - yields each chunk separately."""
            chunk_obj_copy = chunk_obj.copy()
            
            for tool_call_index, tool_call in tool_calls.items():
                try:
                    tool_call_obj = json.loads(repair_json(json.dumps(tool_call)))
                    function_call = ChoiceDeltaFunctionCall(
                        name=tool_call_obj["name"],
                        arguments=tool_call_obj["arguments"]
                    )
                    delta_tool_call = ChoiceDeltaToolCall(
                        index=int(tool_call_index),
                        id=tool_call_obj["id"],
                        function=function_call,
                        type="function"
                    )
                    chunk_obj_copy.choices[0].delta.content = None
                    chunk_obj_copy.choices[0].delta.tool_calls = [delta_tool_call]  
                    chunk_obj_copy.choices[0].finish_reason = "tool_calls"
                    yield f"data: {chunk_obj_copy.json()}\n\n"
                except Exception as e:
                    logger.error(f"Failed to create tool call chunk: {e}")
                    chunk_obj_copy.choices[0].delta.content = None
                    chunk_obj_copy.choices[0].delta.tool_calls = []
                    yield f"data: {chunk_obj_copy.json()}\n\n"
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
                
                async for chunk in response.aiter_bytes():
                    buffer += chunk.decode('utf-8', errors='replace')
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        json_str = _extract_json_data(line)
                        
                        if json_str is None:
                            continue
                            
                        if json_str == '[DONE]':
                            yield 'data: [DONE]\n\n'
                            continue
                        
                        try:
                            chunk_obj = ChatCompletionChunk.parse_raw(json_str)
                            choice = chunk_obj.choices[0]
                            
                            # Handle finish reason - output accumulated tool calls
                            if choice.finish_reason and tool_calls:
                                for tool_call_chunk in _create_tool_call_chunks(tool_calls, chunk_obj):
                                    yield tool_call_chunk

                                yield f"data: [DONE]\n\n"
                                return
                            
                            # Handle tool call deltas
                            if choice.delta.tool_calls:
                                _process_tool_call_delta(choice.delta.tool_calls[0], tool_calls)
                            else:
                                # Regular content chunk
                                yield f"data: {chunk_obj.json()}\n\n"
                                    
                        except Exception as e:
                            logger.error(f"Failed to parse streaming chunk: {e}")
                            # Pass through unparseable data (except ping messages)
                            if not line.strip().startswith(': ping'):
                                yield f"data: {line}\n\n"
                            
                # Process any remaining buffer content
                if buffer.strip():
                    json_str = _extract_json_data(buffer)
                    if json_str and json_str != '[DONE]':
                        try:
                            chunk_obj = ChatCompletionChunk.parse_raw(json_str)
                            yield f"data: {chunk_obj.json()}\n\n"
                        except Exception as e:
                            logger.error(f"Failed to parse trailing chunk: {e}")
                    elif json_str == '[DONE]':
                        yield 'data: [DONE]\n\n'
                            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
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
    async def _ensure_server_running(request_id: str) -> bool:
        """Ensure the AI server is running, reload if necessary."""
        try:
            service_info = get_service_info()
            if "pid" not in service_info:
                logger.info(f"[{request_id}] AI server not running, reloading...")
                return await ServiceHandler.reload_ai_server()
            return True
        except HTTPException:
            logger.info(f"[{request_id}] Service info not available, attempting reload...")
            return False
    
    @staticmethod
    async def process_request(endpoint: str, request_data: Dict[str, Any]):
        """Process a request by adding it to the queue and waiting for the result."""
        request_id = generate_request_id()
        queue_size = RequestProcessor.queue.qsize()
        
        logger.info(f"[{request_id}] Adding request to queue for endpoint {endpoint} (queue size: {queue_size})")
        
        # Update the last request time
        app.state.last_request_time = time.time()
        
        # Check if we need to reload the AI server
        await RequestProcessor._ensure_server_running(request_id)
        
        start_wait_time = time.time()
        future = asyncio.Future()
        await RequestProcessor.queue.put((endpoint, request_data, future, request_id, start_wait_time))
        
        logger.info(f"[{request_id}] Waiting for result from endpoint {endpoint}")
        result = await future
        
        total_time = time.time() - start_wait_time
        logger.info(f"[{request_id}] Request completed for endpoint {endpoint} (total time: {total_time:.2f}s)")
        
        return result
    
    @staticmethod
    async def process_direct(endpoint: str, request_data: Dict[str, Any]):
        """Process a request directly without queueing for administrative endpoints."""
        request_id = generate_request_id()
        logger.info(f"[{request_id}] Processing direct request for endpoint {endpoint}")
        
        app.state.last_request_time = time.time()
        await RequestProcessor._ensure_server_running(request_id)
        
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
    
    @staticmethod
    async def worker():
        """Worker function to process requests from the queue sequentially."""
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
            try:
                service_info = get_service_info()
                if ("pid" in service_info and hasattr(app.state, "last_request_time")):
                    idle_time = time.time() - app.state.last_request_time
                    logger.info(f"Unload checker task, last request time: {app.state.last_request_time}")
                    
                    if idle_time > IDLE_TIMEOUT:
                        logger.info(f"AI server has been idle for {idle_time:.2f}s, unloading...")
                        await ServiceHandler.kill_ai_server()
            except HTTPException:
                pass  # Service info not available, continue checking
            
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
    
    # Kill the AI server if it's running
    try:
        service_info = get_service_info()
        if "pid" in service_info:
            await ServiceHandler.kill_ai_server()
    except HTTPException:
        pass  # Service info not available
    
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
async def update(request: Dict[str, Any]):
    """Update the service information in the app's state."""
    app.state.service_info = request
    return {"status": "ok", "message": "Service info updated successfully"}

# Model-based endpoints that use the request queue
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Endpoint for chat completion requests."""
    if app.state.service_info["task"] == "embed":
        raise HTTPException(status_code=400, detail="Chat completion requests are not supported for embedding models")
    request_dict = convert_request_to_dict(request)
    return await RequestProcessor.process_request("/chat/completions", request_dict)

@app.post("/embeddings")
async def embeddings(request: EmbeddingRequest):
    """Endpoint for embedding requests."""
    request_dict = convert_request_to_dict(request)
    return await RequestProcessor.process_request("/embeddings", request_dict)

@app.post("/v1/chat/completions")
async def v1_chat_completions(request: ChatCompletionRequest):
    """Endpoint for chat completion requests (v1 API)."""
    if app.state.service_info["task"] == "embed":
        raise HTTPException(status_code=400, detail="Chat completion requests are not supported for embedding models")
    request_dict = convert_request_to_dict(request)
    return await RequestProcessor.process_request("/v1/chat/completions", request_dict)

@app.post("/v1/embeddings")
async def v1_embeddings(request: EmbeddingRequest):
    """Endpoint for embedding requests (v1 API)."""
    request_dict = convert_request_to_dict(request)
    return await RequestProcessor.process_request("/v1/embeddings", request_dict)