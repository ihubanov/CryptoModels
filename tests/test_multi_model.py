import asyncio
import os
import pickle
import pytest
import subprocess
import httpx
import time
import json
from unittest import mock
from pathlib import Path
from typing import Dict, Any, List

from fastapi.testclient import TestClient

# Adjust imports based on your actual project structure
from local_ai.core import LocalAIManager, ModelNotFoundError, ServiceStartError
# To test individual FastAPI instances, we need to be careful if 'app' is a global.
# For now, we'll test LocalAIManager's interactions and simulate FastAPI app state.
# from local_ai.apis import app as fastapi_app
from local_ai.schema import ChatCompletionRequest, EmbeddingRequest, Message

# Use a common temporary directory for pickle files if needed, or allow LocalAIManager to use its default
TEST_PICKLE_FILE_NAME = "test_running_service.pkl"

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mocks essential environment variables."""
    monkeypatch.setenv("LLAMA_SERVER", "/fake/path/to/llama-server")
    monkeypatch.setenv("RUNNING_SERVICE_FILE", TEST_PICKLE_FILE_NAME)

@pytest.fixture
def mock_llama_server_path(monkeypatch):
    """Ensures the mocked llama-server path appears to exist."""
    mock_path = mock.MagicMock(spec=Path)
    mock_path.exists.return_value = True
    monkeypatch.setattr("local_ai.core.os.path.exists", lambda p: p == "/fake/path/to/llama-server")


@pytest.fixture
def manager(mock_env_vars, mock_llama_server_path):
    """Provides a LocalAIManager instance with a clean pickle file for each test."""
    pickle_file = Path(TEST_PICKLE_FILE_NAME)
    if pickle_file.exists():
        pickle_file.unlink()

    manager_instance = LocalAIManager()
    # Override the pickle file path for safety if not already done by RUNNING_SERVICE_FILE env var
    manager_instance.pickle_file = pickle_file

    yield manager_instance

    # Teardown: remove the pickle file after the test
    if pickle_file.exists():
        pickle_file.unlink()

# --- Mocks for LocalAIManager dependencies ---

@pytest.fixture
def mock_download_model(monkeypatch):
    async def fake_download_model_from_filecoin_async(hash_str):
        # Simulate model download, return a dummy path
        dummy_model_path = f"/tmp/models/{hash_str}/model.gguf"
        #os.makedirs(os.path.dirname(dummy_model_path), exist_ok=True) # Not needed if path.exists is mocked
        return dummy_model_path

    monkeypatch.setattr(
        "local_ai.core.download_model_from_filecoin_async",
        fake_download_model_from_filecoin_async
    )

@pytest.fixture
def mock_os_path_exists_for_models(monkeypatch):
    """Mocks os.path.exists to return True for dummy model paths."""
    original_exists = os.path.exists
    def new_exists(path_str):
        if str(path_str).startswith("/tmp/models/") and str(path_str).endswith(".gguf"):
            return True
        if str(path_str).startswith("/fake/path/to"): # For llama-server itself
             return True
        return original_exists(path_str) # Call original for other paths (like pickle file)
    monkeypatch.setattr("local_ai.core.os.path.exists", new_exists)
    monkeypatch.setattr("os.path.exists", new_exists) # If used directly in a module

@pytest.fixture
def mock_subprocess_popen(monkeypatch):
    """Mocks subprocess.Popen."""
    mock_processes = []

    class MockPopen:
        def __init__(self, command, stderr, preexec_fn):
            self.command = command
            self.stderr = stderr
            self.preexec_fn = preexec_fn
            self.pid = os.urandom(4).hex() # Generate a fake PID
            self.returncode = None
            mock_processes.append(self)
            logger_name = "ai_process" if "llama-server" in command[0] else "apis_process"
            print(f"MockPopen: Started {logger_name} with PID {self.pid} for command: {' '.join(command)}")

        def poll(self):
            return self.returncode

        def terminate(self):
            print(f"MockPopen: Terminating PID {self.pid}")
            self.returncode = 0
            # In a real scenario, psutil.Process(self.pid).terminate() would be called by manager's stop logic
            # For the mock, we just set returncode.

    monkeypatch.setattr("subprocess.Popen", MockPopen)
    return mock_processes


@pytest.fixture
def mock_wait_for_service(monkeypatch):
    """Mocks LocalAIManager._wait_for_service to always return True."""
    async_mock = mock.AsyncMock(return_value=True)
    # Since _wait_for_service is a method of the class, we patch it where it's used or on the class itself.
    # Patching directly on the class is cleaner if possible.
    monkeypatch.setattr("local_ai.core.LocalAIManager._wait_for_service", async_mock)
    return async_mock

@pytest.fixture
def mock_requests_post(monkeypatch):
    """Mocks requests.post for /update calls and returns the mock for assertions."""
    mock_post_obj = mock.MagicMock()

    def fake_post(url, json, timeout):
        # Simulate the actual call being made through the mock object
        # This allows assertions on mock_post_obj.call_args_list, etc.
        print(f"Mocked requests.post called: URL={url}, JSON content hash (if any): {json.get('hash')}")

        response_mock = mock.MagicMock()
        if "/update" in url:
            response_mock.status_code = 200
            response_mock.raise_for_status = mock.MagicMock() # Mock raise_for_status if called
        else:
            response_mock.status_code = 404 # Default for other posts if any
        return response_mock

    # Assign the fake_post function to the mock object's side_effect attribute
    mock_post_obj.side_effect = fake_post

    # Patch where requests.post is used. Assuming it's used as 'requests.post' in local_ai.core
    monkeypatch.setattr("local_ai.core.requests.post", mock_post_obj)
    return mock_post_obj # Return the MagicMock instance itself


# --- Test Cases ---

@pytest.mark.asyncio
async def test_start_single_model_manager_setup(
    manager: LocalAIManager,
    mock_download_model,
    mock_os_path_exists_for_models,
    mock_subprocess_popen: List[MockPopen], # type hint for clarity
    mock_wait_for_service,
    mock_requests_post # To check /update calls
):
    """Test basic LocalAIManager setup and starting a single model."""
    model_hash1 = "testhash1"

    # Mock metadata fetching for the model if it's done via HTTP
    # For now, assuming metadata is part of what _retry_request_json fetches if file doesn't exist
    with mock.patch.object(manager, '_retry_request_json', return_value={"folder_name": "model_folder_1", "family": "general"}):
        success = manager.start([model_hash1], host="127.0.0.1", context_length=2048)

    assert success, "Manager failed to start the model"
    assert model_hash1 in manager.loaded_models

    model_info = manager.loaded_models[model_hash1]
    assert model_info["hash"] == model_hash1
    assert model_info["model_name"] == "model_folder_1"
    assert "llama_server_port" in model_info
    assert "app_port" in model_info # FastAPI port for this model
    assert "pid" in model_info # llama-server PID
    assert "app_pid" in model_info # FastAPI uvicorn PID

    assert len(mock_subprocess_popen) == 2 # One for llama-server, one for uvicorn

    llama_server_process = next(p for p in mock_subprocess_popen if "llama-server" in p.command[0])
    uvicorn_process = next(p for p in mock_subprocess_popen if "uvicorn" in p.command[0])

    assert str(model_info["llama_server_port"]) in llama_server_process.command
    assert str(model_info["app_port"]) in uvicorn_process.command
    assert model_info["pid"] == llama_server_process.pid
    assert model_info["app_pid"] == uvicorn_process.pid

    # Check if _wait_for_service was called for both llama-server and FastAPI app
    # Expected calls: (llama_server_port), (app_port)
    assert mock_wait_for_service.call_count == 2
    mock_wait_for_service.assert_any_call(manager, model_info["llama_server_port"]) # manager instance is passed as self
    mock_wait_for_service.assert_any_call(manager, model_info["app_port"])

    # Check if requests.post was called for the /update endpoint
    # mock_requests_post is now the MagicMock instance
    expected_update_url = f"http://localhost:{model_info['app_port']}/update"

    # Check if any call matches the expected URL and carries the correct hash in its json payload
    call_found = False
    for call in mock_requests_post.call_args_list:
        args, kwargs = call
        if args[0] == expected_update_url:
            if kwargs.get("json", {}).get("hash") == model_hash1:
                assert kwargs.get("json") == model_info # Ensure the full metadata was sent
                call_found = True
                break
    assert call_found, f"/update endpoint for model {model_hash1} on port {model_info['app_port']} was not called correctly."


@pytest.mark.asyncio
async def test_start_multiple_models(
    manager: LocalAIManager,
    mock_download_model,
    mock_os_path_exists_for_models,
    mock_subprocess_popen: List[mock.MagicMock], # Using MagicMock from mock_subprocess_popen
    mock_wait_for_service,
    mock_requests_post, # MagicMock instance
    monkeypatch # To temporarily change app state for TestClient
):
    """Test LocalAIManager.start() with multiple model hashes."""
    model_hashes = ["testhash1", "testhash2"]
    model_folders = {"testhash1": "model_folder_1", "testhash2": "model_folder_2"}

    # Mock metadata fetching for multiple models
    def mock_retry_json_side_effect(url, retries, delay, timeout):
        for h, folder in model_folders.items():
            if h in url:
                return {"folder_name": folder, "family": "general"}
        return {"folder_name": "unknown_folder", "family": "unknown_family"}

    with mock.patch.object(manager, '_retry_request_json', side_effect=mock_retry_json_side_effect):
        success = manager.start(model_hashes, host="127.0.0.1", context_length=2048)

    assert success, "Manager failed to start multiple models"
    assert len(manager.loaded_models) == len(model_hashes)
    for model_hash in model_hashes:
        assert model_hash in manager.loaded_models

    # Verify distinct ports and correct PIDs
    ports_app = set()
    ports_llama = set()
    pids_app = set()
    pids_llama = set()

    for model_hash in model_hashes:
        model_info = manager.loaded_models[model_hash]
        assert model_info["model_name"] == model_folders[model_hash]
        ports_app.add(model_info["app_port"])
        ports_llama.add(model_info["llama_server_port"])
        pids_app.add(model_info["app_pid"])
        pids_llama.add(model_info["pid"])

    assert len(ports_app) == len(model_hashes), "FastAPI app ports are not unique for each model"
    assert len(ports_llama) == len(model_hashes), "Llama server ports are not unique for each model"
    assert len(pids_app) == len(model_hashes), "FastAPI PIDs are not unique"
    assert len(pids_llama) == len(model_hashes), "Llama PIDs are not unique"

    assert len(mock_subprocess_popen) == 2 * len(model_hashes) # 2 processes per model

    # Verify _wait_for_service calls (2 per model)
    assert mock_wait_for_service.call_count == 2 * len(model_hashes)

    # Verify /update calls for each model
    assert mock_requests_post.call_count == len(model_hashes)
    for model_hash in model_hashes:
        model_info = manager.loaded_models[model_hash]
        expected_update_url = f"http://localhost:{model_info['app_port']}/update"

        call_found = False
        for call in mock_requests_post.call_args_list:
            args, kwargs = call
            if args[0] == expected_update_url and kwargs.get("json", {}).get("hash") == model_hash:
                assert kwargs.get("json") == model_info # Check full metadata
                call_found = True
                break
        assert call_found, f"/update for model {model_hash} on port {model_info['app_port']} not called correctly."

# --- FastAPI App Testing ---
# We need to be able to create a TestClient for an app instance that has its state set.
# This is tricky if 'app' is a global. We might need a factory or careful state manipulation.

# For testing the "One FastAPI per Model" architecture, we'll simulate an instance.
# We need to import the actual FastAPI app from apis.py
from local_ai.apis import app as actual_fastapi_app

def get_test_client_for_model(model_metadata: Dict[str, Any]) -> TestClient:
    """
    Creates a TestClient for a FastAPI app instance, with app.state.service_info
    set to the provided model_metadata.
    This simulates a single FastAPI instance dedicated to one model.
    """
    # It's critical that app.state modifications here don't leak between tests
    # or affect other tests. TestClient(app) usually copies the app.
    # However, app.state might be shared if not handled carefully.
    # A robust way is to have a fixture that provides a fresh app or manages state.
    # For now, we directly set and then clear state if possible, or rely on TestClient isolation.

    # Simulate the state that LocalAIManager would have pushed via /update
    # This is for a *single* model instance.
    actual_fastapi_app.state.service_info = model_metadata
    # Also initialize other relevant states if needed by endpoints
    actual_fastapi_app.state.client = httpx.AsyncClient() # Mock or real if not hitting network
    actual_fastapi_app.state.last_request_time = time.time()

    client = TestClient(actual_fastapi_app)
    return client

def test_v1_models_endpoint_single_instance():
    """
    Tests the /v1/models endpoint for a single FastAPI instance.
    It should only return the model that this instance serves.
    """
    model_metadata = {
        "hash": "model_A_hash",
        "model_name": "model-A",
        "llama_server_port": 8001,
        "app_port": 9001,
        "pid": "pid_A_llama",
        "app_pid": "pid_A_uvicorn",
        "multimodal": False,
        "context_length": 4096,
        "last_activity": time.time(),
        # other fields as expected by /v1/models if any (e.g. created_time)
    }
    client = get_test_client_for_model(model_metadata.copy()) # Use copy to avoid state modification issues

    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()

    assert data["object"] == "list"
    assert len(data["data"]) == 1

    model_entry = data["data"][0]
    assert model_entry["id"] == "model-A"
    assert model_entry["object"] == "model"
    assert "created" in model_entry # Check for presence
    assert model_entry["owned_by"] == "local-ai"

    # Clean up state if directly modified on global app (important if not using app factory)
    if hasattr(actual_fastapi_app.state, "service_info"):
        del actual_fastapi_app.state.service_info


@pytest.mark.asyncio
async def test_chat_completions_routing_to_correct_llama_server():
    """
    Tests that a request to /v1/chat/completions on a specific model's FastAPI instance
    proxies the request to that model's correct llama_server_port.
    """
    model_name_served = "test-model-chat"
    model_metadata = {
        "hash": "chat_hash",
        "model_name": model_name_served,
        "llama_server_port": 8002, # Specific port for this model's llama-server
        "app_port": 9002,
        # ... other necessary fields
    }

    # We need to mock app.state.client.post for the FastAPI instance
    with mock.patch("httpx.AsyncClient.post", new_callable=mock.AsyncMock) as mock_httpx_post:
        # Configure the mock to return a valid-looking response from llama-server
        mock_llama_response = {
            "id": "chatcmpl-mock", "object": "chat.completion", "created": int(time.time()),
            "model": model_name_served, # Llama-server might echo back the model
            "choices": [{"finish_reason": "stop", "index": 0, "message": {"role": "assistant", "content": "Mocked response"}}]}
        mock_httpx_post.return_value = mock.MagicMock(spec=httpx.Response, status_code=200, json=lambda: mock_llama_response, text=json.dumps(mock_llama_response))

        client = get_test_client_for_model(model_metadata.copy())

        chat_request_payload = {
            "model": model_name_served, # Requesting the model this instance serves
            "messages": [{"role": "user", "content": "Hello"}]
        }
        response = client.post("/v1/chat/completions", json=chat_request_payload)

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "Mocked response"

        # Verify httpx.AsyncClient.post was called correctly
        mock_httpx_post.assert_called_once()
        call_args = mock_httpx_post.call_args
        url_called = call_args[0][0] # First positional argument is the URL

        assert f"http://localhost:{model_metadata['llama_server_port']}/v1/chat/completions" == url_called

    # Test mismatch case: request for a different model
    with mock.patch("httpx.AsyncClient.post", new_callable=mock.AsyncMock) as mock_httpx_post_mismatch:
        client_mismatch = get_test_client_for_model(model_metadata.copy()) # Re-init client for clean state if needed

        mismatched_payload = {
            "model": "some-other-model", # Requesting a different model
            "messages": [{"role": "user", "content": "Hello"}]
        }
        response_mismatch = client_mismatch.post("/v1/chat/completions", json=mismatched_payload)

        assert response_mismatch.status_code == 400 # Expecting a client error due to model mismatch
        assert "not served by this API instance" in response_mismatch.json()["detail"]
        mock_httpx_post_mismatch.assert_not_called() # Should not attempt to call llama-server

    # Clean up state
    if hasattr(actual_fastapi_app.state, "service_info"):
        del actual_fastapi_app.state.service_info
    if hasattr(actual_fastapi_app.state, "client"):
        del actual_fastapi_app.state.client

@pytest.mark.asyncio
async def test_embeddings_routing_to_correct_llama_server():
    """
    Tests that a request to /v1/embeddings on a specific model's FastAPI instance
    proxies the request to that model's correct llama_server_port.
    """
    model_name_served = "test-model-embed"
    model_metadata = {
        "hash": "embed_hash",
        "model_name": model_name_served,
        "llama_server_port": 8003, # Specific port for this model's llama-server
        "app_port": 9003,
        # ... other necessary fields
    }

    with mock.patch("httpx.AsyncClient.post", new_callable=mock.AsyncMock) as mock_httpx_post:
        # Configure the mock to return a valid-looking response from llama-server
        mock_llama_response = {
            "object": "list", "data": [{"embedding": [0.1, 0.2], "index": 0, "object": "embedding"}],
            "model": model_name_served
        }
        mock_httpx_post.return_value = mock.MagicMock(spec=httpx.Response, status_code=200, json=lambda: mock_llama_response, text=json.dumps(mock_llama_response))

        client = get_test_client_for_model(model_metadata.copy())

        embedding_request_payload = {
            "model": model_name_served, # Requesting the model this instance serves
            "input": ["Hello world"]
        }
        response = client.post("/v1/embeddings", json=embedding_request_payload)

        assert response.status_code == 200
        assert len(response.json()["data"]) == 1
        assert response.json()["data"][0]["embedding"] == [0.1, 0.2]

        # Verify httpx.AsyncClient.post was called correctly
        mock_httpx_post.assert_called_once()
        call_args = mock_httpx_post.call_args
        url_called = call_args[0][0]

        assert f"http://localhost:{model_metadata['llama_server_port']}/v1/embeddings" == url_called

    # Test mismatch case
    with mock.patch("httpx.AsyncClient.post", new_callable=mock.AsyncMock) as mock_httpx_post_mismatch:
        client_mismatch = get_test_client_for_model(model_metadata.copy())

        mismatched_payload = {
            "model": "some-other-embedding-model",
            "input": ["Hello world"]
        }
        response_mismatch = client_mismatch.post("/v1/embeddings", json=mismatched_payload)

        assert response_mismatch.status_code == 400
        assert "not served by this API instance" in response_mismatch.json()["detail"]
        mock_httpx_post_mismatch.assert_not_called()

    # Clean up state
    if hasattr(actual_fastapi_app.state, "service_info"):
        del actual_fastapi_app.state.service_info
    if hasattr(actual_fastapi_app.state, "client"):
        del actual_fastapi_app.state.client

# --- LocalAIManager stop and persistence tests ---

@pytest.mark.asyncio
async def test_manager_stop_terminates_processes_and_clears_state(
    manager: LocalAIManager,
    mock_download_model,
    mock_os_path_exists_for_models,
    mock_subprocess_popen: List[MockPopen], # MockPopen instances are stored here
    mock_wait_for_service,
    mock_requests_post, # To allow start to complete
    monkeypatch
):
    """Tests that LocalAIManager.stop() terminates all processes and clears state."""
    model_hashes = ["stoptest_hash1", "stoptest_hash2"]
    model_folders = {"stoptest_hash1": "stop_folder_1", "stoptest_hash2": "stop_folder_2"}

    def mock_retry_json_side_effect(url, retries, delay, timeout):
        for h, folder in model_folders.items():
            if h in url:
                return {"folder_name": folder, "family": "general"}
        return {"folder_name": "unknown_folder", "family": "unknown_family"}

    with mock.patch.object(manager, '_retry_request_json', side_effect=mock_retry_json_side_effect):
        manager.start(model_hashes, host="127.0.0.1", context_length=2048)

    assert len(manager.loaded_models) == 2
    original_pids = set()
    for mh in model_hashes:
        original_pids.add(manager.loaded_models[mh]['pid']) # llama-server PID
        original_pids.add(manager.loaded_models[mh]['app_pid']) # FastAPI PID

    assert len(original_pids) == 4 # 2 models, 2 processes each = 4 unique PIDs

    # Mock terminate_process_group to record calls
    mock_terminate_pg = mock.MagicMock(return_value=True)
    monkeypatch.setattr(manager, "terminate_process_group", mock_terminate_pg)

    # Mock os.remove for checking pickle file removal
    mock_os_remove = mock.MagicMock()
    monkeypatch.setattr("os.remove", mock_os_remove)

    manager.stop()

    # Assert terminate_process_group was called for all original PIDs
    assert mock_terminate_pg.call_count == len(original_pids)
    called_pids_with_terminate = {call_args[0][0] for call_args in mock_terminate_pg.call_args_list} # PID is the first arg
    assert called_pids_with_terminate == original_pids

    assert not manager.loaded_models, "loaded_models should be empty after stop"
    mock_os_remove.assert_called_once_with(manager.pickle_file)


@pytest.mark.asyncio
async def test_manager_persistence_load_and_update(
    mock_env_vars, # To set TEST_PICKLE_FILE_NAME for both managers
    mock_llama_server_path,
    mock_download_model,
    mock_os_path_exists_for_models,
    mock_subprocess_popen,
    mock_wait_for_service,
    mock_requests_post # This will capture /update calls from both managers
):
    """Tests dumping state to pickle, loading by a new manager, and /update calls on load."""
    manager1 = LocalAIManager() # Uses TEST_PICKLE_FILE_NAME due to mock_env_vars

    model_hashes = ["persist_hash1", "persist_hash2"]
    model_folders = {"persist_hash1": "persist_folder_1", "persist_hash2": "persist_folder_2"}

    def mock_retry_json_side_effect(url, retries, delay, timeout):
        for h, folder in model_folders.items():
            if h in url:
                return {"folder_name": folder, "family": "general"}
        return None

    with mock.patch.object(manager1, '_retry_request_json', side_effect=mock_retry_json_side_effect):
        manager1.start(model_hashes, host="127.0.0.1", context_length=2048)

    assert len(manager1.loaded_models) == 2
    original_loaded_models_manager1 = manager1.loaded_models.copy()

    # manager1.start() already calls _dump_running_service()
    assert manager1.pickle_file.exists()

    # Clear previous calls to mock_requests_post from manager1.start()
    mock_requests_post.reset_mock()

    # Create a new manager - this should load from pickle and call /update
    manager2 = LocalAIManager()
    assert len(manager2.loaded_models) == 2

    # Verify that loaded_models are the same
    for model_hash in model_hashes:
        assert model_hash in manager2.loaded_models
        # Compare key details, PIDs might change if we were actually restarting real processes
        assert manager2.loaded_models[model_hash]["model_name"] == original_loaded_models_manager1[model_hash]["model_name"]
        assert manager2.loaded_models[model_hash]["app_port"] == original_loaded_models_manager1[model_hash]["app_port"]
        assert manager2.loaded_models[model_hash]["llama_server_port"] == original_loaded_models_manager1[model_hash]["llama_server_port"]

    # Verify that manager2's __init__ called /update for each loaded model
    # These are new /update calls during manager2's initialization
    assert mock_requests_post.call_count == len(model_hashes)
    for model_hash in model_hashes:
        model_info = manager2.loaded_models[model_hash]
        expected_update_url = f"http://localhost:{model_info['app_port']}/update"

        update_call_found = False
        for call in mock_requests_post.call_args_list:
            args, kwargs = call
            if args[0] == expected_update_url and kwargs.get("json", {}).get("hash") == model_hash:
                # During __init__ update, the full service_metadata is sent
                assert kwargs.get("json") == model_info
                update_call_found = True
                break
        assert update_call_found, f"/update for reloaded model {model_hash} on port {model_info['app_port']} was not called by manager2.__init__."

    # Clean up pickle file explicitly (though fixture also does it)
    if manager1.pickle_file.exists():
        manager1.pickle_file.unlink()
    if manager2.pickle_file.exists(): # Should be the same file
        manager2.pickle_file.unlink()


# --- Tests for resource path resolution ---

@mock.patch("local_ai.core.os.path.exists")
@mock.patch("local_ai.core.importlib.util.find_spec")
def test_resource_path_resolution(
    mock_find_spec,
    mock_os_exists,
    manager: LocalAIManager # Use the existing manager fixture
):
    """Tests _get_model_template_path and _get_model_best_practice_path resolution."""

    # --- Scenario 1: Successful resolution ---
    # Configure find_spec mock
    mock_spec = mock.MagicMock()
    # Simulate a realistic site-packages path
    mock_spec.origin = "/path/to/venv/lib/python3.9/site-packages/local_ai/__init__.py"
    mock_find_spec.return_value = mock_spec

    # Configure os.path.exists mock for this scenario
    # It should return True only for the correctly constructed paths
    expected_template_path = "/path/to/venv/lib/python3.9/site-packages/local_ai/examples/templates/gemma.jinja"
    expected_best_practice_path = "/path/to/venv/lib/python3.9/site-packages/local_ai/examples/best_practices/gemma.json"

    def side_effect_os_exists(path_to_check):
        if path_to_check == expected_template_path:
            return True
        if path_to_check == expected_best_practice_path:
            return True
        return False
    mock_os_exists.side_effect = side_effect_os_exists

    # Test template path
    template_path = manager._get_model_template_path("gemma")
    assert template_path == expected_template_path
    mock_find_spec.assert_called_with("local_ai")
    mock_os_exists.assert_called_with(expected_template_path)

    # Test best practice path
    best_practice_path = manager._get_model_best_practice_path("gemma")
    assert best_practice_path == expected_best_practice_path
    mock_find_spec.assert_called_with("local_ai") # Called again
    mock_os_exists.assert_called_with(expected_best_practice_path)

    mock_find_spec.reset_mock()
    mock_os_exists.reset_mock() # Reset for next scenario

    # --- Scenario 2: File not found by os.path.exists ---
    mock_find_spec.return_value = mock_spec # find_spec still works
    mock_os_exists.side_effect = None # Clear previous side_effect
    mock_os_exists.return_value = False # All os.path.exists calls return False

    template_path_notfound = manager._get_model_template_path("gemma")
    assert template_path_notfound is None
    # (Optional: Check for logger.warning call if logging is testable)

    best_practice_path_notfound = manager._get_model_best_practice_path("gemma")
    assert best_practice_path_notfound is None
    # (Optional: Check for logger.warning)

    mock_find_spec.reset_mock()
    mock_os_exists.reset_mock()

    # --- Scenario 3: Package spec not found (importlib.util.find_spec returns None) ---
    mock_find_spec.return_value = None # Simulate package not found

    template_path_no_spec = manager._get_model_template_path("gemma")
    assert template_path_no_spec is None
    # (Optional: Check for logger.error)

    best_practice_path_no_spec = manager._get_model_best_practice_path("gemma")
    assert best_practice_path_no_spec is None
    # (Optional: Check for logger.error)

    mock_find_spec.reset_mock() # Clean up

    # --- Scenario 4: Spec origin is None ---
    mock_spec_no_origin = mock.MagicMock()
    mock_spec_no_origin.origin = None
    mock_find_spec.return_value = mock_spec_no_origin

    template_path_no_origin = manager._get_model_template_path("gemma")
    assert template_path_no_origin is None
    # (Optional: Check for logger.error)

    best_practice_path_no_origin = manager._get_model_best_practice_path("gemma")
    assert best_practice_path_no_origin is None
    # (Optional: Check for logger.error)
