import os
import json
import sys
import pty
import re
import aiohttp
import asyncio
import psutil
import time
import shutil
import random
import subprocess
import tempfile
from pathlib import Path
from loguru import logger
from huggingface_hub import HfApi
from huggingface_hub import snapshot_download
from eternal_zoo.models import FEATURED_MODELS, HASH_TO_MODEL
from eternal_zoo.utils import compute_file_hash, async_extract_zip, async_move, async_rmtree
from eternal_zoo.constants import DEFAULT_MODEL_DIR, POSTFIX_MODEL_PATH, GATEWAY_URLS, ETERNAL_AI_METADATA_GW, PREFIX_DOWNLOAD_LOG

SLEEP_TIME = 2
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
if psutil:
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
        if progress_tracker:
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
            logger.info("[ProgressTracker]")
            logger.info(f"{PREFIX_DOWNLOAD_LOG} "
                        f"--progress {percentage:.1f}% ({self.completed_files}/{self.total_files} files) "
                        f"--speed {speed_mbps:.2f} MB/s")

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
    

async def fetch_model_metadata_async(filecoin_hash: str, max_attempts: int = 10) -> tuple[bool, dict | None]:
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
    # for meta-data add more with ETERNAL_AI_METADATA_GW
    gateways_with_eternal = GATEWAY_URLS + [ETERNAL_AI_METADATA_GW]
    best_gateway = await pick_fastest_gateway(filecoin_hash, gateways_with_eternal)
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


async def download_model_async_by_hash(hf_data: dict, filecoin_hash: str | None = None) -> tuple[bool, str | None]:
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

    try:
        # Fetch model metadata using the dedicated async function
        success, data = await fetch_model_metadata_async(filecoin_hash)
        if not success:
            logger.error("Failed to fetch model metadata")
            return False, None
            
        data["filecoin_hash"] = filecoin_hash
        data["local_path"] = local_path_str
        is_lora = data.get("lora", False)
        lora_metadata = {}

        if is_lora:
            # First, try to load metadata from local LoRA model if it exists
            if os.path.exists(local_path_str):
                metadata_path = os.path.join(local_path_str, "metadata.json")
                try:
                    with open(metadata_path, "r") as f:
                        lora_metadata = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    logger.warning(f"Failed to load local LoRA metadata: {e}")
                    lora_metadata = {}
            
                base_model_hash = lora_metadata["base_model"]
                lora_base_model_path_str = str(DEFAULT_MODEL_DIR / f"{base_model_hash}{POSTFIX_MODEL_PATH}")
                if os.path.exists(lora_base_model_path_str):
                    logger.info(f"Using existing LoRA model at {local_path_str}")
                    logger.info(f"Using existing base model at {lora_base_model_path_str}")
                    return True, local_path_str
                else:
                    logger.warning(f"LoRA model exists but base model not found at {lora_base_model_path_str}")
                    logger.info(f"Downloading missing base model: {base_model_hash}")
                    base_model_hf_data = None
                    if base_model_hash in HASH_TO_MODEL:
                        base_model_hf_data = FEATURED_MODELS[HASH_TO_MODEL[base_model_hash]]
                    success, base_model_path = await download_model_async_by_hash(base_model_hf_data, base_model_hash)
                    if not success:
                        logger.error(f"Failed to download base model: {base_model_hash}")
                        return False, None
                    logger.info(f"Successfully downloaded base model and using existing LoRA model")
                    return True, local_path_str
        else:
            if os.path.exists(local_path_str):
                logger.info(f"Using existing model at {local_path_str}")
                return True, local_path_str

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

        if is_lora:
            success, hf_res = await download_model_from_hf(hf_data)
            if not success:
                logger.error("Failed to download LoRA model, falling back to Filecoin")
                return False, None
            if hf_res["is_folder"]:
                await async_move(hf_res["model_path"], local_path_str)
                await async_rmtree(hf_res["model_path"])
            lora_metadata_path = os.path.join(local_path_str, "metadata.json")
            try:
                with open(lora_metadata_path, "r") as f:
                    lora_metadata = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.error(f"Failed to load LoRA metadata from downloaded model: {e}")
                return False, None
            
            base_model_hash = lora_metadata.get("base_model", None)
            if base_model_hash is None:
                logger.error("No base_model found in LoRA metadata")
                return False, None

            base_model_hf_data = None

            if base_model_hash in HASH_TO_MODEL:
                base_model_hf_data = FEATURED_MODELS[HASH_TO_MODEL[base_model_hash]]

            print(f"base_model_hf_data: {base_model_hf_data}")
            print(f"base_model_hash: {base_model_hash}")

            success, base_model_path = await download_model_async_by_hash(base_model_hf_data, base_model_hash)
            if not success:
                logger.error(f"Failed to download base model: {base_model_hash}")
                return False, None
            logger.info(f"Successfully downloaded LoRA model and base model: {local_path_str}")
            return True, local_path_str
        else:
            success, hf_res = await download_model_from_hf(hf_data)
            if success:
                logger.info(f"Successfully downloaded: {hf_res}")
                if hf_res["is_folder"]:
                    await async_move(hf_res["model_path"], local_path_str)
                    await async_rmtree(hf_res["model_path"])
                else:
                    await async_move(hf_res["model_path"], local_path_str)
                return True, local_path_str
            else:
                logger.info("Download failed, falling back to Filecoin")
        
        # Prepare for Filecoin download
        folder_path = Path(tempfile.mkdtemp(prefix=f"filecoin_download_{data['folder_name']}_"))

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
    
# Cache for repository sizes to avoid repeated API calls
_repo_size_cache = {}

def get_repo_size(repo_id: str) -> int:
    """Get repository size with caching to avoid repeated API calls"""
    if repo_id in _repo_size_cache:
        return _repo_size_cache[repo_id]
    
    try:
        api = HfApi()
        repo_info = api.model_info(repo_id=repo_id, files_metadata=True)
        total_size_bytes = 0
        for sibling in repo_info.siblings:
            total_size_bytes += sibling.size or 0
        _repo_size_cache[repo_id] = total_size_bytes
        return total_size_bytes
    except Exception as e:
        logger.warning(f"Failed to get repo size for {repo_id}: {e}")
        return 0

def clear_repo_size_cache():
    """Clear the repository size cache"""
    global _repo_size_cache
    _repo_size_cache.clear()
    logger.debug("Repository size cache cleared")

def calculate_folder_size_fast(folder_path: Path) -> int:
    """Calculate folder size using os.walk for better performance, excluding .cache directories"""
    total_size = 0
    try:
        if folder_path.exists() and folder_path.is_dir():
            for dirpath, dirnames, filenames in os.walk(str(folder_path)):
                for filename in filenames:
                    try:
                        file_path = Path(dirpath) / filename
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
                    except (OSError, PermissionError):
                        # Skip files we can't access
                        continue
    except Exception as e:
        logger.warning(f"Error calculating folder size for {folder_path}: {e}")
    return total_size

# Keep the original function for backward compatibility
def calculate_folder_size(folder_path: Path) -> int:
    """Calculate the total size of a folder and all its contents in bytes"""
    return calculate_folder_size_fast(folder_path)

class HuggingFaceProgressTracker:
    """Track HuggingFace download progress by monitoring actual folder size with optimizations"""
    
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.start_time = time.time()
        self.is_running = True
        self.lock = asyncio.Lock()  # Use asyncio.Lock instead of threading.Lock
        self.last_log_time = self.start_time
        self.watch_dir = None

        # Get total repository size with caching
        self.total_size_bytes = get_repo_size(repo_id) * random.uniform(1.1, 1.5)

        self.total_size_mb = self.total_size_bytes / (1024 * 1024)
        print(self.total_size_mb)
        
        # Progress tracking with caching
        self.progress_task = None
        self.last_folder_size = 0
        self.last_size_check_time = 0
        self.size_cache_duration = 2.0  # Cache folder size for 2 seconds
        
        # Batch progress updates to reduce logging overhead
        self.pending_progress_updates = 0
        self.last_progress_log_time = 0
        self.progress_log_interval = 5.0  # Log progress every 5 seconds
        
        # Download progress
        self.last_speed_mbps = 0.0
        self.last_percentage = 0.0

    def set_watch_dir(self, watch_dir: str):
        self.watch_dir = watch_dir
        
    def start_progress_tracking(self):
        """Start the background progress tracking task"""
        if self.progress_task is None:
            self.progress_task = asyncio.create_task(self._periodic_progress_update())

    def get_current_progress(self) -> tuple[float, float]:
        """Get current download progress and speed with caching"""
        if self.watch_dir is None:
            return self.last_percentage, self.last_speed_mbps
        
        current_time = time.time()
        
        # Cache folder size checks to avoid excessive filesystem operations
        if current_time - self.last_size_check_time >= self.size_cache_duration:
            try:
                self.last_folder_size = calculate_folder_size_fast(Path(self.watch_dir))
                print(self.last_folder_size)
                self.last_size_check_time = current_time
            except Exception as e:
                logger.debug(f"Error calculating folder size: {e}")
                # Use last known size if calculation fails
                pass
        
        percentage = (self.last_folder_size / self.total_size_bytes * 100) if self.total_size_bytes > 0 else 0
        percentage = min(percentage, 100.0)  # Cap at 100%
        
        elapsed_time = current_time - self.start_time
        current_speed_mbps = (self.last_folder_size / (1024 * 1024)) / elapsed_time if elapsed_time > 0 else 0

        percentage = max(self.last_percentage, percentage)
        current_speed_mbps = max(self.last_speed_mbps, current_speed_mbps)

        self.last_percentage = percentage
        self.last_speed_mbps = current_speed_mbps
        
        return percentage, current_speed_mbps

    async def _periodic_progress_update(self):
        """Background task to log progress periodically with optimized frequency"""
        while self.is_running:
            try:
                await asyncio.sleep(1.0)  # Check every second
                
                current_time = time.time()
                # Log progress every 5 seconds as requested
                if current_time - self.last_progress_log_time >= self.progress_log_interval:
                    async with self.lock:
                        self.last_progress_log_time = current_time
                        percentage, current_speed_mbps = self.get_current_progress()
                        logger.info(
                            f"{PREFIX_DOWNLOAD_LOG} "
                            f"--progress {percentage:.1f}% "
                            f"--speed {current_speed_mbps:.2f} MB/s "
                        )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in HF progress update task: {e}")
                
    def complete_download(self):
        """Mark download as completed"""
        async def _complete():
            async with self.lock:
                self.is_running = False
                elapsed_time = time.time() - self.start_time
                final_folder_size = calculate_folder_size_fast(Path(self.watch_dir)) if self.watch_dir else 0
                actual_speed_mbps = (final_folder_size / (1024 * 1024)) / elapsed_time if elapsed_time > 0 else 0
                
                logger.info("[HuggingFaceProgressTracker]")
                logger.info(
                    f"{PREFIX_DOWNLOAD_LOG} "
                    f"--progress 100.0% "
                    f"--speed {actual_speed_mbps:.2f} MB/s "
                    f"--final_size {final_folder_size / (1024 * 1024):.2f} MB"
                )
        
        # Run the async completion in the event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_complete())
            else:
                loop.run_until_complete(_complete())
        except RuntimeError:
            # If no event loop is running, just log without async
            elapsed_time = time.time() - self.start_time
            final_folder_size = calculate_folder_size_fast(Path(self.watch_dir)) if self.watch_dir else 0
            actual_speed_mbps = (final_folder_size / (1024 * 1024)) / elapsed_time if elapsed_time > 0 else 0
            
            logger.info("[HuggingFaceProgressTracker]")
            logger.info(
                f"{PREFIX_DOWNLOAD_LOG} "
                f"--progress 100.0% "
                f"--speed {actual_speed_mbps:.2f} MB/s "
                f"--final_size {final_folder_size / (1024 * 1024):.2f} MB"
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

async def download_model_from_hf(data: dict, output_dir: Path | None = None) -> tuple[bool, dict | None]:
    """
    Download model from HuggingFace Hub with infinite retry logic and exponential backoff.
    Downloads forever until finished or canceled by user.
    
    Args:
        data: Model metadata dictionary containing hf_repo, hf_file, and local_path
        
    Returns:
        tuple[bool, str | None]: (success, local_path) - True and path if successful, False and None if failed
    """

    res = {}
    repo_id = data["repo"]
    model = data.get("model", None)
    projector = data.get("projector", None)
    pattern = data.get("pattern", None)
    model_dir = str(output_dir) if output_dir else str(DEFAULT_MODEL_DIR/f"tmp_{repo_id.replace('/', '_')}")
        
    attempt = 1
    while True:  # Infinite loop until success or user cancellation
        try:
            logger.info(f"Download attempt {attempt} for {repo_id}")
                        
            # Run the synchronous download in a thread executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            if model is None:
                progress_tracker = HuggingFaceProgressTracker(repo_id)
                progress_tracker.start_progress_tracking()

                progress_tracker.set_watch_dir(model_dir)
    
                if pattern:
                    # Download only the files that match the allow_patterns
                    await loop.run_in_executor(
                        None,
                        lambda: snapshot_download(
                            repo_id=repo_id,
                            local_dir=model_dir,
                            allow_patterns=[f"*{pattern}*"],
                            token=os.getenv("HF_TOKEN")                       
                            
                        )
                    )
                    res["is_folder"] = True
                    res["model_path"] = os.path.join(model_dir, pattern)
                
                else:

                    await loop.run_in_executor(
                        None,
                        lambda: snapshot_download(
                            repo_id=repo_id,
                            local_dir=model_dir,
                            token=os.getenv("HF_TOKEN")  
                        )                     
                    )
                    res["is_folder"] = True
                    res["model_path"] = model_dir
            else:
                # For single file downloads, we can't easily track progress by folder size
                # since the file is downloaded directly to output_dir, not a temp folder
                # The PTY-based download already provides progress output

                await loop.run_in_executor(
                    None,
                    lambda: run_hf_download_with_pty(repo_id, model, model_dir, token= os.getenv("HF_TOKEN", None))
                )

                res["is_folder"] = False
                res["model_path"] = os.path.join(model_dir, model)

                # Download projector if specified
                if projector:
                    final_projector_path = os.path.join(model_dir, model + "-projector")
                    if os.path.exists(final_projector_path):
                        res["projector_path"] = final_projector_path
                        logger.info(f"Projector file {final_projector_path} already exists")
                    else:
                        await loop.run_in_executor(
                            None,
                            lambda: run_hf_download_with_pty(repo_id, projector, model_dir, token= os.getenv("HF_TOKEN", None))
                        )
                        await async_move(os.path.join(model_dir, projector), final_projector_path)
                        res["projector_path"] = final_projector_path
            
            logger.info(f"Successfully downloaded model: {model_dir}")
            return True, res
        
        except KeyboardInterrupt:
            logger.info("Download canceled by user")

        except Exception as e:
            logger.warning(f"Download attempt {attempt} failed: {e}")
            
            # Calculate exponential backoff with jitter
            backoff_time = _calculate_backoff(attempt)
            logger.warning(f"Retrying in {backoff_time}s (attempt {attempt + 1})")
            await asyncio.sleep(backoff_time)
            
            # Create a new progress tracker for the next attempt
            progress_tracker = HuggingFaceProgressTracker(repo_id)
            attempt += 1

def run_hf_download_with_pty(repo_id: str, model: str, local_dir: str, token: str | None = None):
    """
    Run HuggingFace download in a subprocess with PTY, capturing and logging all output (including progress bar).
    Only works on Linux/macOS.
    """

    if token:
        script = f'''
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="{repo_id}", filename="{model}", local_dir="{local_dir}", token="{token}")
'''
    else:
        script = f'''
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="{repo_id}", filename="{model}", local_dir="{local_dir}")
'''

    master_fd, slave_fd = pty.openpty()
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        universal_newlines=True
    )
    os.close(slave_fd)
    buffer = ""
    progress_re = re.compile(
        r"([0-9]+)%\s+([0-9.]+[GMK]?)/([0-9.]+[GMK]?)\s*\[.*?([0-9.]+)MB/s\]"  # Simple format
        r"|([0-9]+)%\|.*?\|\s*([0-9.]+[GMK]?)/([0-9.]+[GMK]?)\s*\[.*?([0-9.]+)MB/s\]"  # Bar format
    )
    while True:
        try:
            data = os.read(master_fd, 1024).decode()
            if not data:
                break
            buffer += data
            while '\r' in buffer or '\n' in buffer:
                if '\r' in buffer:
                    line, buffer = buffer.split('\r', 1)
                else:
                    line, buffer = buffer.split('\n', 1)
                if line.strip():
                    match = progress_re.search(line)
                    if match:
                        if match.group(1):  # Simple format
                            percent = float(match.group(1))
                            speed = float(match.group(4))
                            downloaded = match.group(2)
                            total = match.group(3)
                        else:  # Bar format
                            percent = float(match.group(5))
                            speed = float(match.group(8))
                            downloaded = match.group(6)
                            total = match.group(7)
                        logger.info(f"[HF-PTY] {line}")
                        logger.info(f"{PREFIX_DOWNLOAD_LOG} --progress {percent:.2f}% --speed {speed:.2f} MB/s --size {downloaded}/{total}")
                    else:
                        logger.info(f"[HF-PTY] {line}")
        except OSError:
            break
    process.wait()
    os.close(master_fd)
    if process.returncode != 0:
        raise Exception(f"HuggingFace download failed with return code {process.returncode}")
    
async def download_model_async(hf_data: dict, filecoin_hash: str | None = None) -> tuple[bool, str | None]:
    logger.info(f"Downloading model: {hf_data}")
    logger.info(f"Filecoin hash: {filecoin_hash}")
    path = None
    if filecoin_hash:
        success, path = await download_model_async_by_hash(hf_data, filecoin_hash)
        if not success:
            logger.error("Failed to download model")
            return False, None
    else:
        final_dir = DEFAULT_MODEL_DIR
        model = hf_data.get("model", None)

        if model is None:
            final_dir = final_dir / hf_data["repo"].replace("/", "_")

        success, hf_res = await download_model_from_hf(hf_data, final_dir)
        if not success:
            logger.error("Failed to download model")
            return False, None
        path = final_dir
    return True, path
