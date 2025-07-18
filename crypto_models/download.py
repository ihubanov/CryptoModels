import os
import json
import random
import aiohttp
import asyncio
import psutil
import time
import shutil
import threading
from pathlib import Path
from loguru import logger
from crypto_models.models import MODELS
from huggingface_hub import hf_hub_download, snapshot_download
from crypto_models.constants import DEFAULT_MODEL_DIR, POSTFIX_MODEL_PATH, GATEWAY_URLS
from crypto_models.utils import compute_file_hash, async_extract_zip, async_move, async_rmtree

SLEEP_TIME = 5
# Set MAX_ATTEMPTS: if only one gateway, try at least 6 times; otherwise, try 3 times per gateway
if len(GATEWAY_URLS) == 1:
    MAX_ATTEMPTS = 6
else:
    MAX_ATTEMPTS = len(GATEWAY_URLS) * 3

# Extraction buffer factor for disk space estimation
EXTRACTION_BUFFER_FACTOR = 2  # 2x the total download size

# Performance optimizations (dynamic based on system RAM and CPU)
total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
cpu_cores = os.cpu_count() or 4

# Set chunk size dynamically based on RAM
if total_ram_gb >= 16:
    CHUNK_SIZE_MB = 16
elif total_ram_gb >= 8:
    CHUNK_SIZE_MB = 8
else:
    CHUNK_SIZE_MB = 4

# Set flush interval dynamically based on RAM
if total_ram_gb >= 8:
    FLUSH_INTERVAL_MB = 128
else:
    FLUSH_INTERVAL_MB = 50

print(f"[CONFIG] CHUNK_SIZE_MB={CHUNK_SIZE_MB}, FLUSH_INTERVAL_MB={FLUSH_INTERVAL_MB}, RAM={total_ram_gb:.1f}GB, CPU={cpu_cores}")

PROGRESS_BATCH_SIZE = 10 * 1024 * 1024  # Batch progress updates for 10MB chunks
# Set MAX_CONCURRENT_DOWNLOADS dynamically based on CPU cores and available RAM (capped at 32)
cpu_limit = cpu_cores * 2
ram_limit = 32  # Default if psutil is not available
if psutil is not None:
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    ram_limit = int(total_ram_gb * 4)  # Estimate: 4 downloads per GB RAM
MAX_CONCURRENT_DOWNLOADS = min(32, cpu_limit, ram_limit)
CONNECTION_POOL_SIZE = 32  # Increased connection pool

# Add file logger for download operations
logger.add("download.log", rotation="10 MB", retention="10 days", encoding="utf-8")


async def _cleanup_on_failure(folder_path: Path, local_path_str: str) -> None:
    """
    Clean up temporary files and folders on download failure.
    
    Args:
        folder_path: Path to the temporary extraction folder
        local_path_str: Path to the partially downloaded model
    """
    try:
        # Clean up temporary folder
        if folder_path and folder_path.exists():
            logger.info(f"Cleaning up temporary folder: {folder_path}")
            await async_rmtree(str(folder_path))
        
        # Clean up partial model files
        partial_paths = [
            Path(local_path_str),
            Path(local_path_str + "-projector"),
            Path(local_path_str + ".tmp")
        ]
        
        for path in partial_paths:
            if path.exists():
                logger.info(f"Removing partial file: {path}")
                path.unlink(missing_ok=True)
                
    except Exception as e:
        logger.warning(f"Failed to clean up temporary files: {e}")


def _calculate_backoff(attempt: int, base_sleep: int = SLEEP_TIME, max_backoff: int = 300) -> int:
    """
    Calculate exponential backoff with jitter.
    
    Args:
        attempt: Current attempt number (1-based)
        base_sleep: Base sleep time in seconds
        max_backoff: Maximum backoff time in seconds
        
    Returns:
        int: Backoff time in seconds
    """
    import random
    
    # Exponential backoff with jitter
    backoff = min(base_sleep * (2 ** (attempt - 1)), max_backoff)
    # Add random jitter (±20%)
    jitter = random.uniform(0.8, 1.2)
    return int(backoff * jitter)

async def download_single_file_async(session: aiohttp.ClientSession, file_info: dict, folder_path: Path,
                                     max_attempts: int = MAX_ATTEMPTS, progress_callback=None,
                                     progress_tracker=None) -> tuple:
    """
    Asynchronously download a single file and verify its SHA256 hash, with retries.

    Args:
        session (aiohttp.ClientSession): Reusable HTTP session.
        file_info (dict): Contains 'cid', 'file_hash', and 'file_name'.
        folder_path (Path): Directory to save the file.
        max_attempts (int): Number of retries on failure.
        progress_callback: Optional callback for progress updates.

    Returns:
        tuple: (Path to file if successful, None) or (None, error message).
    """
    cid = file_info["cid"]
    expected_hash = file_info["file_hash"]
    file_name = file_info["file_name"]
    file_path = folder_path / file_name
    attempts = 0

    # Use a simple temp file for atomic operations
    temp_path = folder_path / f"{file_name}.tmp"

    # Check if file already exists with correct hash
    if file_path.exists():
        try:
            computed_hash = compute_file_hash(file_path)
            if computed_hash == expected_hash:
                logger.info(f"File {cid} already exists with correct hash.")
                
                # Account for cached files in progress tracking
                if progress_tracker:
                    # Get file size from file_info (in MB) and convert to bytes
                    file_size_mb = file_info.get("size_mb", 0)
                    if file_size_mb > 0:
                        file_size_bytes = int(file_size_mb * 1024 * 1024)
                        # Add to expected size
                        await progress_tracker.add_file_size(file_size_bytes)
                        # Add to total downloaded (but not actually downloaded - this is cached)
                        await progress_tracker.add_cached_file_size(file_size_bytes)
                        logger.info(f"Added cached file size to progress: {file_size_mb} MB")
                
                return file_path, None
            else:
                logger.warning(f"File {cid} exists but hash mismatch. Retrying...")
                file_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Error checking existing file {cid}: {e}")
            file_path.unlink(missing_ok=True)

    # Clean up any existing temp file
    if temp_path.exists():
        temp_path.unlink(missing_ok=True)

    while attempts < max_attempts:
        # For each attempt, pick the fastest gateway at this moment
        fastest_gateway = await pick_fastest_gateway(cid, GATEWAY_URLS)
        gateways_order = [fastest_gateway] + [gw for gw in GATEWAY_URLS if gw != fastest_gateway]
        # Use the fastest gateway for the first attempt, then round robin through the rest
        gateway = gateways_order[attempts % len(gateways_order)]
        url = f"{gateway}{cid}"
        # Print which URL will be used for downloading
        logger.info(f"[download_single_file_async] Attempt {attempts+1}/{max_attempts}: Downloading from URL: {url} ---> {file_path}")
        # Update the current URL and file name in the progress tracker
        if progress_tracker is not None:
            progress_tracker.current_url = url
            progress_tracker.current_file_name = file_name

        try:
            # Use optimized chunk size for faster downloads
            chunk_size = CHUNK_SIZE_MB * 1024 * 1024  # 4MB chunks

            # Use a longer timeout for large files
            timeout = aiohttp.ClientTimeout(total=900, connect=120, sock_read=300, sock_connect=120)

            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    # Get total size for progress tracking
                    total_size = int(response.headers.get("content-length", 0))

                    # Report file size to progress tracker
                    if progress_tracker and total_size > 0:
                        await progress_tracker.add_file_size(total_size)

                    # Download to temp file
                    with temp_path.open("wb") as f:
                        # Add timeout protection for each chunk
                        last_data_time = asyncio.get_event_loop().time()
                        chunk_timeout = 180  # 180 seconds without data is a timeout
                        bytes_written = 0

                        # Download with per-chunk timeout protection
                        async for chunk in response.content.iter_chunked(chunk_size):
                            # Reset timeout timer when data is received
                            last_data_time = asyncio.get_event_loop().time()

                            # Write chunk
                            f.write(chunk)
                            bytes_written += len(chunk)

                            # Report progress to overall tracker with batched updates
                            if progress_callback:
                                await progress_callback(len(chunk))

                            # Flush to disk less frequently for better performance
                            if bytes_written >= FLUSH_INTERVAL_MB * 1024 * 1024:  # 50MB
                                f.flush()
                                os.fsync(f.fileno())
                                bytes_written = 0

                            # Check if download has been idle
                            current_time = asyncio.get_event_loop().time()
                            if current_time - last_data_time > chunk_timeout:
                                raise asyncio.TimeoutError(f"No data received for {chunk_timeout} seconds")

                        # Final flush
                        f.flush()
                        os.fsync(f.fileno())

                    # Verify hash
                    computed_hash = compute_file_hash(temp_path)
                    if computed_hash == expected_hash:
                        # Rename temp file to final file only after successful verification
                        if file_path.exists():
                            file_path.unlink()
                        temp_path.rename(file_path)
                        logger.info(f"File {cid} downloaded and verified successfully.")
                        return file_path, None
                    else:
                        logger.warning(f"Hash mismatch for {cid}. Expected {expected_hash}, got {computed_hash}.")
                        # Clean up temp file on hash mismatch
                        temp_path.unlink(missing_ok=True)
                else:
                    logger.warning(f"Failed to download {cid}. Status: {response.status}")
                    # For certain status codes, we might want to fail faster
                    if response.status in (404, 403, 401):
                        wait_time = SLEEP_TIME
                    else:
                        wait_time = min(SLEEP_TIME * (2 ** attempts), 300)

        except asyncio.TimeoutError:
            logger.warning(f"Timeout downloading {cid}")
            # Clean up temp file on timeout
            temp_path.unlink(missing_ok=True)
            wait_time = min(SLEEP_TIME * (2 ** attempts), 300)
        except aiohttp.ClientError as e:
            logger.warning(f"Client error downloading {cid}: {e}")
            temp_path.unlink(missing_ok=True)
            wait_time = min(SLEEP_TIME * (2 ** attempts), 300)
        except Exception as e:
            logger.warning(f"Exception downloading {cid}: {e}")
            temp_path.unlink(missing_ok=True)
            wait_time = min(SLEEP_TIME * (2 ** attempts), 300)

        attempts += 1
        if attempts < max_attempts:
            logger.info(f"Retrying in {wait_time}s (Attempt {attempts + 1}/{max_attempts})")
            await asyncio.sleep(wait_time)
        else:
            logger.warning(f"Failed to download {cid} after {max_attempts} attempts.")
            # Clean up temp file on final failure
            temp_path.unlink(missing_ok=True)
            return None, f"Failed to download {cid} after {max_attempts} attempts."
    return None, ""

class ProgressTracker:
    """Track download progress across multiple concurrent downloads with batched updates"""

    def __init__(self, total_files: int, filecoin_hash: str):
        self.total_files = total_files
        self.filecoin_hash = filecoin_hash
        self.total_bytes_downloaded = 0  # Total bytes (cached + downloaded)
        self.total_bytes_actually_downloaded = 0  # Only bytes actually downloaded from network
        self.total_bytes_expected = 0
        self.completed_files = 0
        self.start_time = time.time()
        self.last_log_time = 0
        self.lock = asyncio.Lock()

        # Batched progress tracking to reduce lock contention
        self.pending_bytes = 0
        self.pending_lock = asyncio.Lock()

        # Track the current downloading URL and file name
        self.current_url = None
        self.current_file_name = None

        # Start background task for periodic progress updates
        self.progress_task = asyncio.create_task(self._periodic_progress_update())

    async def add_file_size(self, file_size: int):
        """Add expected file size to total"""
        async with self.lock:
            self.total_bytes_expected += file_size

    async def add_cached_file_size(self, file_size: int):
        """Add cached file size to total downloaded (but not actually downloaded)"""
        async with self.lock:
            self.total_bytes_downloaded += file_size
            # Don't add to total_bytes_actually_downloaded since this wasn't downloaded

    async def update_progress_batched(self, bytes_downloaded: int):
        """Update progress using batched approach to reduce lock contention"""
        # Use a separate lock for pending bytes to minimize contention
        async with self.pending_lock:
            self.pending_bytes += bytes_downloaded

            # Only acquire main lock if we have enough pending bytes
            if self.pending_bytes >= PROGRESS_BATCH_SIZE:
                pending_to_process = self.pending_bytes
                self.pending_bytes = 0

                # Quick update to main counter
                async with self.lock:
                    self.total_bytes_downloaded += pending_to_process
                    self.total_bytes_actually_downloaded += pending_to_process

    # Compatibility method for backward compatibility
    async def update_progress(self, bytes_downloaded: int):
        """Backward compatibility wrapper"""
        await self.update_progress_batched(bytes_downloaded)

    async def _periodic_progress_update(self):
        """Background task to log progress periodically"""
        while True:
            try:
                await asyncio.sleep(1.0)  # Check every second

                # Process any remaining pending bytes
                async with self.pending_lock:
                    if self.pending_bytes > 0:
                        pending_to_process = self.pending_bytes
                        self.pending_bytes = 0

                        async with self.lock:
                            self.total_bytes_downloaded += pending_to_process
                            self.total_bytes_actually_downloaded += pending_to_process

                # Log progress every 2 seconds
                current_time = time.time()
                if current_time - self.last_log_time >= 2.0:
                    async with self.lock:
                        self.last_log_time = current_time
                        elapsed_time = current_time - self.start_time
                        # Use only actually downloaded bytes for speed calculation
                        speed_mbps = (self.total_bytes_actually_downloaded / (
                                1024 * 1024)) / elapsed_time if elapsed_time > 0 else 0

                        # Calculate percentage based on bytes downloaded vs total expected
                        if self.total_bytes_expected > 0:
                            percentage = (self.total_bytes_downloaded / self.total_bytes_expected) * 100
                            percentage = min(percentage, 100.0)  # Cap at 100%
                        else:
                            # Fall back to file-based progress if no total size available
                            percentage = (self.completed_files / self.total_files) * 100

                        logger.info(
                            f"[CRYPTOAGENTS_LOGGER] [MODEL_INSTALL] "
                            f"--progress {percentage:.1f}% ({self.completed_files}/{self.total_files} files) "
                            f"--speed {speed_mbps:.2f} MB/s "
                        )
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in progress update task: {e}")

    async def complete_file(self):
        """Mark a file as completed"""
        async with self.lock:
            self.completed_files += 1
            elapsed_time = time.time() - self.start_time
            # Use only actually downloaded bytes for speed calculation
            speed_mbps = (self.total_bytes_actually_downloaded / (1024 * 1024)) / elapsed_time if elapsed_time > 0 else 0

            # Calculate percentage based on bytes downloaded vs total expected
            if self.total_bytes_expected > 0:
                percentage = (self.total_bytes_downloaded / self.total_bytes_expected) * 100
                percentage = min(percentage, 100.0)  # Cap at 100%
            else:
                # Fall back to file-based progress if no total size available
                percentage = (self.completed_files / self.total_files) * 100

            logger.info(
                f"[CRYPTOAGENTS_LOGGER] [MODEL_INSTALL] --progress {percentage:.1f}% ({self.completed_files}/{self.total_files} files) --speed {speed_mbps:.2f} MB/s")

    async def cleanup(self):
        """Clean up background tasks"""
        if hasattr(self, 'progress_task'):
            self.progress_task.cancel()
            try:
                await self.progress_task
            except asyncio.CancelledError:
                pass


async def download_files_from_lighthouse_async(data: dict) -> list:
    """
    Asynchronously download files concurrently using Filecoin CIDs and verify hashes.

    Args:
        data (dict): JSON data with 'folder_name', 'files', 'num_of_files', and 'filecoin_hash'.

    Returns:
        list: Paths of successfully downloaded files, or empty list if failed.
    """
    folder_name = data["folder_name"]
    folder_path = Path(folder_name)
    folder_path.mkdir(exist_ok=True)
    num_of_files = data["num_of_files"]
    filecoin_hash = data["filecoin_hash"]
    files = data["files"]

    # Calculate total size for progress indication
    total_files = len(files)

    # Use optimized concurrency limits
    max_concurrent_downloads = min(MAX_CONCURRENT_DOWNLOADS, max(4, num_of_files))
    semaphore = asyncio.Semaphore(max_concurrent_downloads)

    # Create progress tracker
    progress_tracker = ProgressTracker(total_files, filecoin_hash)

    # Wrapper for download with semaphore
    async def download_with_semaphore(session, file_info, folder_path):
        async with semaphore:
            return await download_single_file_async(
                session, file_info, folder_path,
                progress_callback=progress_tracker.update_progress_batched,
                progress_tracker=progress_tracker
            )

    # Use larger connection pool for better performance
    connector = aiohttp.TCPConnector(limit=CONNECTION_POOL_SIZE, ssl=False)
    timeout = aiohttp.ClientTimeout(total=None, connect=120, sock_connect=120, sock_read=300)

    logger.info(f"Downloading {total_files} files with max {max_concurrent_downloads} concurrent downloads")

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Create tasks
        tasks = [
            download_with_semaphore(session, file_info, folder_path)
            for file_info in files
        ]

        # Track overall progress
        successful_downloads = []
        failed_downloads = []

        # Use as_completed to process files as they complete
        for i, future in enumerate(asyncio.as_completed(tasks), 1):
            try:
                path, error = await future
                if path:
                    successful_downloads.append(path)
                    await progress_tracker.complete_file()
                else:
                    failed_downloads.append(error)
                    logger.warning(f"Download failed: {error}")
            except Exception as e:
                logger.warning(f"Unexpected error in download task: {e}")
                failed_downloads.append(str(e))

        # Clean up progress tracker
        await progress_tracker.cleanup()

        # Check if all downloads were successful
        if len(successful_downloads) == num_of_files:
            logger.info(f"All {num_of_files} files downloaded successfully.")
            return successful_downloads
        else:
            logger.warning(f"Downloaded {len(successful_downloads)} out of {num_of_files} files.")
            if failed_downloads:
                logger.warning(f"Failed downloads ({len(failed_downloads)}):")
                for i, error in enumerate(failed_downloads[:5], 1):
                    logger.warning(f"  {i}. {error}")
                if len(failed_downloads) > 5:
                    logger.warning(f"  ... and {len(failed_downloads) - 5} more errors")
            # return successful_downloads if successful_downloads else []
            raise Exception("Failed downloads")


async def pick_fastest_gateway(filecoin_hash: str, gateways: list[str], timeout: int = 5) -> str:
    """
    Check the speed of each gateway and return the fastest one for the given filecoin_hash.
    If only one gateway is provided, return it immediately.
    Args:
        filecoin_hash (str): The IPFS hash to test download speed for.
        gateways (list[str]): List of gateway URLs.
        timeout (int): Timeout in seconds for each speed test.
    Returns:
        str: The fastest gateway URL, or the first in the list if all fail.
    """

    # If there is only one gateway, return it immediately (no need to check)
    if len(gateways) == 1:
        logger.info(f"[pick_fastest_gateway] ✅ Only one gateway provided, returning: {gateways[0]}")
        return gateways[0]

    logger.info(f"[pick_fastest_gateway] 🚦 Checking speed for {len(gateways)} gateways with hash: {filecoin_hash}")

    async def check_gateway(gateway: str) -> tuple[str, float]:
        url = f"{gateway}{filecoin_hash}"
        logger.info(f"[pick_fastest_gateway] 🔍 Testing gateway: {url}")
        start = asyncio.get_event_loop().time()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                # Use GET with Range header to fetch only the first 1KB, since some gateways may not support HEAD
                headers = {"Range": "bytes=0-1023"}
                async with session.get(url, headers=headers) as resp:
                    if resp.status in (200, 206):  # 206 = Partial Content
                        elapsed = asyncio.get_event_loop().time() - start
                        logger.info(f"[pick_fastest_gateway] ⏱️ Gateway {gateway} responded in {elapsed:.3f} seconds.")
                        return gateway, elapsed
                    else:
                        logger.warning(f"[pick_fastest_gateway] ⚠️ Gateway {gateway} returned status {resp.status}.")
        except Exception as e:
            logger.warning(f"[pick_fastest_gateway] ❌ Error with gateway {gateway}: {e}")
        # Return infinity if the gateway is not available or too slow
        return gateway, float('inf')

    # Run all gateway checks concurrently
    tasks = [check_gateway(gw) for gw in gateways]
    results = await asyncio.gather(*tasks)
    # Filter for gateways that responded successfully
    valid_results = [r for r in results if r[1] != float('inf')]
    if valid_results:
        fastest = min(valid_results, key=lambda x: x[1])
        logger.info(f"[pick_fastest_gateway] 🚀 Fastest gateway selected: {fastest[0]} (time: {fastest[1]:.3f} seconds)\n")
        return fastest[0]
    else:
        logger.warning(f"[pick_fastest_gateway] ⚠️ All gateways timed out or failed. Using the first gateway as fallback: {gateways[0]}")
        return gateways[0]


# Helper function to check disk space

def check_disk_space(path: Path, required_bytes: int = 2 * 1024 * 1024 * 1024) -> None:
    """
    Check if there is enough free disk space at the given path.
    
    Args:
        path (Path): Directory to check.
        required_bytes (int): Minimum free space required in bytes (default: 2GB).
        
    Raises:
        Exception: If not enough disk space, with detailed space information.
    """
    try:
        total, used, free = shutil.disk_usage(str(path))
        
        if free < required_bytes:
            required_gb = required_bytes / (1024 ** 3)
            free_gb = free / (1024 ** 3)
            total_gb = total / (1024 ** 3)
            used_gb = used / (1024 ** 3)
            
            logger.error(f"Disk space check failed at {path}")
            logger.error(f"Total: {total_gb:.2f} GB, Used: {used_gb:.2f} GB, Free: {free_gb:.2f} GB")
            logger.error(f"Required: {required_gb:.2f} GB, Shortfall: {(required_bytes - free) / (1024 ** 3):.2f} GB")
            
            raise Exception(f"Not enough disk space: required {required_gb:.2f} GB, available {free_gb:.2f} GB")
        
        # Log successful disk space check for debugging
        free_gb = free / (1024 ** 3)
        required_gb = required_bytes / (1024 ** 3)
        logger.debug(f"Disk space check passed: {free_gb:.2f} GB available, {required_gb:.2f} GB required")
        
    except OSError as e:
        logger.error(f"Failed to check disk space at {path}: {e}")
        raise Exception(f"Unable to check disk space at {path}: {e}")
    

async def fetch_model_metadata_async(filecoin_hash: str, max_attempts: int = 3) -> tuple[bool, dict | None]:
    """
    Asynchronously fetch model metadata from the fastest gateway with retry logic.
    
    Args:
        filecoin_hash (str): The IPFS hash of the model
        max_attempts (int): Maximum number of retry attempts
        
    Returns:
        dict: Model metadata JSON, or None if all attempts fail
    """
    metadata_path = DEFAULT_MODEL_DIR / f"{filecoin_hash}.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return True, json.load(f)
    
    # Select the fastest gateway before downloading
    logger.info("Checking gateway speeds...")
    best_gateway = await pick_fastest_gateway(filecoin_hash, GATEWAY_URLS)
    logger.info(f"Using fastest gateway: {best_gateway}")
    input_link = f"{best_gateway}{filecoin_hash}"

    # Set up session parameters with optimized limits
    timeout = aiohttp.ClientTimeout(total=180, connect=60)
    connector = aiohttp.TCPConnector(limit=CONNECTION_POOL_SIZE, ssl=False)

    # Use exponential backoff for retries
    for attempt in range(1, max_attempts + 1):
        backoff = min(SLEEP_TIME * (2 ** (attempt - 1)), 300)

        try:
            logger.info(f"Downloading model metadata (attempt {attempt}/{max_attempts})")

            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                async with session.get(input_link) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch metadata: HTTP {response.status}")
                        if attempt < max_attempts:
                            logger.warning(f"Retrying in {backoff} seconds")
                            await asyncio.sleep(backoff)
                            continue
                        else:
                            return False, None

                    # Parse metadata
                    data = await response.json()
                    data["filecoin_hash"] = filecoin_hash
                    with open(metadata_path, "w") as f:
                        json.dump(data, f)
                    return True, data

        except aiohttp.ClientError as e:
            logger.warning(f"HTTP error on attempt {attempt}: {e}")
            if attempt < max_attempts:
                logger.warning(f"Retrying in {backoff} seconds")
                await asyncio.sleep(backoff)
                continue
            else:
                raise Exception(f"HTTP error after {max_attempts} attempts: {e}")
        except Exception as e:
            logger.warning(f"Download attempt {attempt} failed: {e}")
            if attempt < max_attempts:
                logger.warning(f"Retrying in {backoff} seconds")
                await asyncio.sleep(backoff)
                continue
            else:
                return False, None

    logger.error("All metadata fetch attempts failed")
    return False, None


async def download_model_async(filecoin_hash: str) -> tuple[bool, str | None]:
    """
    Asynchronously download a model from Filecoin using its IPFS hash.
    This function will select the fastest gateway from a list of gateways by testing their response times,
    and use the fastest one to download the model.
    
    Args:
        filecoin_hash (str): The IPFS hash of the model
        
    Returns:
        tuple[bool, str | None]: (success, local_path) - True and path if successful, False and None if failed
    """
    # Ensure output directory exists
    DEFAULT_MODEL_DIR.mkdir(exist_ok=True, parents=True)
    local_path = DEFAULT_MODEL_DIR / f"{filecoin_hash}{POSTFIX_MODEL_PATH}"
    local_path_str = str(local_path.absolute())

    # Early return if model already exists
    if os.path.exists(local_path_str):
        logger.info(f"Using existing model at {local_path_str}")
        return True, local_path_str

    try:
        # Fetch model metadata using the dedicated async function
        success, data = await fetch_model_metadata_async(filecoin_hash)
        if not success:
            logger.error("Failed to fetch model metadata")
            return False, None
            
        data["filecoin_hash"] = filecoin_hash
        data["local_path"] = local_path_str
        model_metadata = MODELS.get(filecoin_hash, None)

        # More accurate disk space check after metadata is fetched
        if "files" in data:
            total_size_mb = sum(f.get("size_mb", 512) for f in data["files"])
            data["total_size_mb"] = total_size_mb
            required_space_bytes = int(total_size_mb * EXTRACTION_BUFFER_FACTOR * 1024 * 1024)
            try:
                check_disk_space(DEFAULT_MODEL_DIR, required_bytes=required_space_bytes)
            except Exception as e:
                logger.error(f"Insufficient disk space for model extraction: {e}")
                return False, None
            
        if model_metadata is not None:

            data["repo"] = model_metadata["repo"]
            data["model"] = model_metadata.get("model", None)
            data["projector"] = model_metadata.get("projector", None)
            is_lora = data.get("lora", False)
            if is_lora:
                success, hf_local_path = await download_model_from_hf(data)
                if not success:
                    logger.error("Failed to download LoRA model, falling back to Filecoin")
                else:
                    metadata_path = os.path.join(hf_local_path, "metadata.json")
                    metadata = {}
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        base_model_hash = metadata["base_model"]
                        success, base_model_path = await download_model_async(base_model_hash)
                        if not success:
                            logger.error(f"Failed to download base model: {base_model_hash}")
                            return False, None
                        logger.info(f"Successfully downloaded LoRA model and base model: {hf_local_path}")
                        return True, hf_local_path
                    except Exception as e:
                        logger.error(f"Failed to load metadata: {e}")
                        return False, None
            else:
                success, hf_local_path = await download_model_from_hf(data)
                if success:
                    logger.info(f"Successfully downloaded from HuggingFace: {hf_local_path}")
                    return True, hf_local_path
                else:
                    logger.info("Download failed, falling back to Filecoin")
        
        # Prepare for Filecoin download
        folder_path = Path.cwd() / data["folder_name"]
        folder_path.mkdir(exist_ok=True, parents=True)

        # Download and process model with exponential backoff
        for attempt in range(1, MAX_ATTEMPTS + 1):
            backoff = _calculate_backoff(attempt)
            
            try:
                logger.info(f"Download attempt {attempt}/{MAX_ATTEMPTS}")
                
                # Download files from Filecoin
                paths = await download_files_from_lighthouse_async(data)
                if not paths:
                    if attempt < MAX_ATTEMPTS:
                        logger.warning(f"Download failed, retrying in {backoff}s")
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        logger.error("All download attempts failed")
                        await _cleanup_on_failure(folder_path, local_path_str)
                        return False, None

                # Extract downloaded files
                try:
                    logger.info("Extracting downloaded files...")
                    await async_extract_zip(paths)
                except Exception as e:
                    # Don't retry if disk space issue
                    if "Not enough disk space" in str(e):
                        logger.error(f"Extraction failed due to insufficient disk space: {e}")
                        await _cleanup_on_failure(folder_path, local_path_str)
                        return False, None
                    
                    if attempt < MAX_ATTEMPTS:
                        logger.warning(f"Extraction failed, retrying in {backoff}s: {e}")
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        logger.error(f"Extraction failed after {MAX_ATTEMPTS} attempts: {e}")
                        await _cleanup_on_failure(folder_path, local_path_str)
                        return False, None

                # Move files to final location
                final_path = await _move_model_to_final_location(data, folder_path, local_path_str)
                if final_path:
                    logger.info(f"Model download complete: {final_path}")
                    return True, final_path
                
                # If move failed, retry
                if attempt < MAX_ATTEMPTS:
                    logger.warning(f"Move failed, retrying in {backoff}s")
                    await asyncio.sleep(backoff)
                    continue
                else:
                    logger.error(f"Move failed after {MAX_ATTEMPTS} attempts")
                    await _cleanup_on_failure(folder_path, local_path_str)
                    return False, None

            except aiohttp.ClientError as e:
                logger.warning(f"HTTP error on attempt {attempt}: {e}")
                if attempt < MAX_ATTEMPTS:
                    logger.warning(f"Retrying in {backoff}s")
                    await asyncio.sleep(backoff)
                    continue
                else:
                    logger.error(f"HTTP error after {MAX_ATTEMPTS} attempts: {e}")
                    await _cleanup_on_failure(folder_path, local_path_str)
                    return False, None
                    
            except Exception as e:
                # Don't retry if disk space issue
                if "Not enough disk space" in str(e):
                    logger.error(f"Download failed due to insufficient disk space: {e}")
                    await _cleanup_on_failure(folder_path, local_path_str)
                    return False, None
                
                logger.warning(f"Download attempt {attempt} failed: {e}")
                if attempt < MAX_ATTEMPTS:
                    logger.warning(f"Retrying in {backoff}s")
                    await asyncio.sleep(backoff)
                    continue
                else:
                    logger.error(f"Download failed after {MAX_ATTEMPTS} attempts: {e}")
                    await _cleanup_on_failure(folder_path, local_path_str)
                    return False, None

    except Exception as e:
        logger.error(f"Download failed: {e}")
        # Clean up on unexpected failure
        try:
            await _cleanup_on_failure(folder_path, local_path_str)
        except:
            pass  # Ignore cleanup errors
        return False, None


async def _move_model_to_final_location(data: dict, folder_path: Path, local_path_str: str) -> str | None:
    """
    Move extracted model files to their final location.
    
    Args:
        data: Model metadata dictionary
        folder_path: Path to the temporary extraction folder
        local_path_str: Final destination path for the model
        
    Returns:
        str | None: Final model path if successful, None if failed
    """
    try:
        source_text_path = folder_path / data["folder_name"]
        source_text_path = source_text_path.absolute()
        
        if not source_text_path.exists():
            logger.error(f"Model not found at {source_text_path}")
            return None

        # Handle projector path for multimodal models
        source_projector_path = folder_path / (data["folder_name"] + "-projector")
        source_projector_path = source_projector_path.absolute()

        if source_projector_path.exists():
            projector_dest = local_path_str + "-projector"
            logger.info(f"Moving projector to {projector_dest}")
            await async_move(str(source_projector_path), projector_dest)

        # Move model to final location
        logger.info(f"Moving model to {local_path_str}")
        await async_move(str(source_text_path), local_path_str)

        # Clean up temporary folder after successful move
        if folder_path.exists():
            logger.info(f"Cleaning up temporary folder {folder_path}")
            await async_rmtree(str(folder_path))

        return local_path_str
        
    except Exception as e:
        logger.error(f"Failed to move model: {e}")
        return None
    

class HuggingFaceProgressTracker:
    """Track HuggingFace download progress with simulated progress updates"""
    
    def __init__(self, total_size_mb: float, repo_id: str):
        self.total_size_mb = total_size_mb
        self.total_size_bytes = int(total_size_mb * 1024 * 1024)
        self.repo_id = repo_id
        self.downloaded_bytes = 0
        self.start_time = time.time()
        self.last_log_time = 0
        self.is_running = True
        self.lock = threading.Lock()
        
        # Start background task for periodic progress updates
        self.progress_task = None
        
    def start_progress_tracking(self):
        """Start the background progress tracking task"""
        if self.progress_task is None:
            self.progress_task = asyncio.create_task(self._periodic_progress_update())
        
    async def _periodic_progress_update(self):
        """Background task to log progress periodically every 5 seconds"""
        while self.is_running:
            try:
                await asyncio.sleep(1.0)  # Check every second
                
                current_time = time.time()
                estimated_speed_mbps = random.uniform(2.0, 6.0)
                # Log progress every 5 seconds as requested
                if current_time - self.last_log_time >= 5.0:
                    with self.lock:
                        self.last_log_time = current_time
                        elapsed_time = current_time - self.start_time
                        
                        # Simulate progress based on estimated speed
                        if elapsed_time > 0:
                            simulated_downloaded = estimated_speed_mbps * elapsed_time * 1024 * 1024
                            self.downloaded_bytes = int(simulated_downloaded)
                            
                            # Calculate percentage but cap at 95% until download is actually complete
                            percentage = (self.downloaded_bytes / self.total_size_bytes) * 100
                            percentage = min(percentage, 95.0)  # Cap at 95% during download
                            
                            # Calculate current speed
                            current_speed_mbps = (self.downloaded_bytes / (1024 * 1024)) / elapsed_time
                            
                            logger.info(
                                f"[CRYPTOAGENTS_LOGGER] [MODEL_INSTALL] "
                                f"--progress {percentage:.1f}% "
                                f"--speed {current_speed_mbps:.2f} MB/s "
                            )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in HF progress update task: {e}")
                
    def complete_download(self):
        """Mark download as completed"""
        with self.lock:
            self.downloaded_bytes = self.total_size_bytes
            self.is_running = False
            elapsed_time = time.time() - self.start_time
            actual_speed_mbps = (self.total_size_bytes / (1024 * 1024)) / elapsed_time if elapsed_time > 0 else 0
            
            logger.info(
                f"[CRYPTOAGENTS_LOGGER] [MODEL_INSTALL] "
                f"--progress 100.0% "
                f"--speed {actual_speed_mbps:.2f} MB/s "
            )
            
    async def cleanup(self):
        """Clean up background tasks"""
        self.is_running = False
        if self.progress_task:
            self.progress_task.cancel()
            try:
                await self.progress_task
            except asyncio.CancelledError:
                pass


async def download_model_from_hf(data: dict, max_attempts: int = 3) -> tuple[bool, str | None]:
    """
    Download model from HuggingFace Hub with retry logic and progress tracking.
    
    Args:
        data: Model metadata dictionary containing hf_repo, hf_file, and local_path
        max_attempts: Maximum number of retry attempts
        
    Returns:
        tuple[bool, str | None]: (success, local_path) - True and path if successful, False and None if failed
    """
    
    repo_id = data["repo"]
    model = data["model"]
    projector = data["projector"]
    local_path_str = data["local_path"]
    tmp_model_dir = str(DEFAULT_MODEL_DIR / f"tmp_{time.time()}")
    
    # Get total size for progress tracking
    total_size_mb = data.get("total_size_mb", 1000)  # Default to 1GB if not specified
    
    # Create progress tracker
    progress_tracker = HuggingFaceProgressTracker(total_size_mb, repo_id)
    
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"Download attempt {attempt}/{max_attempts} for {repo_id}")
            
            # Start progress tracking
            progress_tracker.start_progress_tracking()
            
            # Run the synchronous download in a thread executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            if model is None:
                # Download entire repository
                await loop.run_in_executor(
                    None,
                    lambda: snapshot_download(
                        repo_id=repo_id,
                        local_dir=tmp_model_dir
                    )
                )
                await async_move(tmp_model_dir, local_path_str)
            else:
                # Download specific model file
                await loop.run_in_executor(
                    None,
                    lambda: hf_hub_download(
                        repo_id=repo_id,
                        filename=model,
                        local_dir=DEFAULT_MODEL_DIR
                    )
                )
                await async_move(str(DEFAULT_MODEL_DIR / model), local_path_str)
                
                # Download projector if specified
                if projector is not None:
                    projector_path = local_path_str + "-projector"
                    await loop.run_in_executor(
                        None,
                        lambda: hf_hub_download(
                            repo_id=repo_id,
                            filename=projector,
                            local_dir=DEFAULT_MODEL_DIR
                        )
                    )
                    await async_move(str(DEFAULT_MODEL_DIR / projector), projector_path)
            
            # Mark download as completed
            progress_tracker.complete_download()
            await progress_tracker.cleanup()
            
            logger.info(f"Successfully downloaded model from HuggingFace: {local_path_str}")
            return True, local_path_str
            
        except Exception as e:
            logger.warning(f"Download attempt {attempt} failed: {e}")
            
            # Clean up progress tracker on failure
            await progress_tracker.cleanup()
            
            if attempt < max_attempts:
                logger.warning(f"Retrying in {SLEEP_TIME}s")
                await asyncio.sleep(SLEEP_TIME)
                # Create a new progress tracker for the next attempt
                progress_tracker = HuggingFaceProgressTracker(total_size_mb, repo_id)
            else:
                logger.error(f"Download failed after {max_attempts} attempts")
                return False, None
    
    return False, None

if __name__ == "__main__":
    model_hash = "bafkreihq4usl2t3i6pqoilvorp4up263yieuxcqs6xznlmrig365bvww5i"

    data = asyncio.run(download_model_async(model_hash))
    print(data)