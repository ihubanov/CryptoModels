import sys
import os
import asyncio
import argparse
import random
import json
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

from eternal_zoo.version import __version__
from eternal_zoo.config import DEFAULT_CONFIG
from eternal_zoo.manager import EternalZooManager
from eternal_zoo.upload import upload_folder_to_lighthouse
from eternal_zoo.constants import DEFAULT_MODEL_DIR, POSTFIX_MODEL_PATH
from eternal_zoo.models import HASH_TO_MODEL, FEATURED_MODELS, MODEL_TO_HASH
from eternal_zoo.download import download_model_async, fetch_model_metadata_async

manager = EternalZooManager()

def get_all_downloaded_models() -> list:
    """
    Get all downloaded model hashes from the llms-storage directory.

    Returns:
        list: List of model hashes that have been downloaded
    """
    downloaded_models = []

    if not DEFAULT_MODEL_DIR.exists():
        return downloaded_models

    # Look for all .gguf files in the directory
    for model_file in DEFAULT_MODEL_DIR.glob(f"*.json"):
        model_hash = model_file.stem
        if model_hash:  # Make sure it's not empty
            downloaded_models.append(model_hash)

    return downloaded_models

def print_banner():
    """Display a beautiful banner for the CLI"""
    console = Console()
    banner_text = """
███████╗████████╗███████╗██████╗ ███╗   ██╗ █████╗ ██╗          ███████╗ ██████╗  ██████╗
██╔════╝╚══██╔══╝██╔════╝██╔══██╗████╗  ██║██╔══██╗██║          ╚══███╔╝██╔═══██╗██╔═══██╗
█████╗     ██║   █████╗  ██████╔╝██╔██╗ ██║███████║██║            ███╔╝ ██║   ██║██║   ██║
██╔══╝     ██║   ██╔══╝  ██╔══██╗██║╚██╗██║██╔══██║██║           ███╔╝  ██║   ██║██║   ██║
███████╗   ██║   ███████╗██║  ██║██║ ╚████║██║  ██║███████╗     ███████╗╚██████╔╝╚██████╔╝
╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝     ╚══════╝ ╚═════╝  ╚═════╝
"""

    panel = Panel(
        Text(banner_text, style="bold cyan", justify="center"),
        title=f"[bold green]Eternal Zoo CLI v{__version__}[/bold green]",
        subtitle="[italic]Peer-to-Peer AI Model Management[/italic]",
        border_style="bright_blue",
        padding=(0, 0)
    )
    console.print(panel)

def print_success(message):
    """Print success message with styling"""
    rprint(f"[bold green]✅ {message}[/bold green]")

def print_error(message):
    """Print error message with styling"""
    rprint(f"[bold red]❌ {message}[/bold red]")

def print_info(message):
    """Print info message with styling"""
    rprint(f"[bold blue]ℹ️  {message}[/bold blue]")

def print_warning(message):
    """Print warning message with styling"""
    rprint(f"[bold yellow]⚠️  {message}[/bold yellow]")

def show_available_models():
    """Display available models"""
    for model_name in FEATURED_MODELS:
        print(f"  {model_name}")

class CustomHelpFormatter(argparse.HelpFormatter):
    """Custom help formatter for better styling"""
    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super()._format_action_invocation(action)
        default = super()._format_action_invocation(action)
        return f"[bold cyan]{default}[/bold cyan]"

def parse_args():
    """Parse command line arguments with beautiful help formatting"""
    parser = argparse.ArgumentParser(
        description="🚀 Eternal Zoo - Peer-to-Peer AI Model Management Tool",
        formatter_class=CustomHelpFormatter,
        epilog="💡 For more information, visit: https://github.com/eternalai-org/eternal-zoo"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"eternal-zoo v{__version__} 🎉"
    )

    subparsers = parser.add_subparsers(
        dest='command',
        help="🛠️  Available commands for managing AI models",
        metavar="COMMAND"
    )

    # Model command group
    model_command = subparsers.add_parser(
        "model",
        help="🤖 Model management operations",
        description="Manage your decentralized AI models"
    )
    model_subparsers = model_command.add_subparsers(
        dest='model_command',
        help="Model operations",
        metavar="OPERATION"
    )

    # Model run command
    run_command = model_subparsers.add_parser(
        "run",
        help="🚀 Launch AI model server with multi-model support",
        description="Start serving models locally with multi-model and on-demand loading support"
    )

    run_command.add_argument(
        "model_name",
        nargs='?',
        help="🏷️  Model name(s) - single: qwen3-1.7b or multi: qwen3-14b,qwen3-4b (first is main, others on-demand)"
    )
    run_command.add_argument(
        "--hash",
        type=str,
        help="🔗 Comma-separated Filecoin hashes (alternative to model names)",
        metavar="HASH1,HASH2,..."
    )
    run_command.add_argument(
        "--hf-repo",
        type=str,
        help="🤗 Hugging Face model repository",
        metavar="REPO"
    )
    run_command.add_argument(
        "--hf-file",
        type=str,
        help="🤗 Hugging Face model file",
        metavar="FILE"
    )
    run_command.add_argument(
        "--mmproj",
        type=str,
        help="🔍 Multimodal Projector File",
        metavar="MMProj"
    )
    run_command.add_argument(
        "--mmproj-url",
        type=str,
        help="🌐 URL to a multimodal projector file",
        metavar="URL"
    )
    run_command.add_argument(
        "--port",
        type=int,
        default=DEFAULT_CONFIG.network.DEFAULT_PORT,
        help=f"🌐 Port number for the server (default: {DEFAULT_CONFIG.network.DEFAULT_PORT})",
        metavar="PORT"
    )
    run_command.add_argument(
        "--host",
        type=str,
        default=DEFAULT_CONFIG.network.DEFAULT_HOST,
        help=f"🏠 Host address for the server (default: {DEFAULT_CONFIG.network.DEFAULT_HOST})",
        metavar="HOST"
    )
    run_command.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONFIG.model.DEFAULT_CONTEXT_LENGTH,
        help=f"📏 Context length for the model (default: {DEFAULT_CONFIG.model.DEFAULT_CONTEXT_LENGTH})",
        metavar="LENGTH"
    )
    run_command.add_argument(
        "--task",
        type=str,
        default="chat",
        choices=["chat", "embed", "image-generation", "image-edit"],
        help="🎯 Model task type (default: chat)",
        metavar="TYPE"
    )
    run_command.add_argument(
        "--config-name",
        type=str,
        default=None,
        choices=["flux-dev", "flux-schnell"],
        help="🔍 Model config name (default: None), need for image-generation and image-edit models",
        metavar="CONFIG"
    )

    # # Model serve command
    # serve_command = model_subparsers.add_parser(
    #     "serve",
    #     help="🎯 Serve all downloaded models with optional main model selection",
    #     description="Run all models in llms-storage with a main model (randomly selected if not specified)"
    # )
    # serve_command.add_argument(
    #     "--main-hash",
    #     help="🔗 Hash of the main model to serve (if not specified, uses random model)",
    #     metavar="HASH"
    # )
    # serve_command.add_argument(
    #     "--port",
    #     type=int,
    #     default=config.network.DEFAULT_PORT,
    #     help=f"🌐 Port number for the server (default: {config.network.DEFAULT_PORT})",
    #     metavar="PORT"
    # )
    # serve_command.add_argument(
    #     "--host",
    #     type=str,
    #     default=config.network.DEFAULT_HOST,
    #     help=f"🏠 Host address for the server (default: {config.network.DEFAULT_HOST})",
    #     metavar="HOST"
    # )
    # serve_command.add_argument(
    #     "--context-length",
    #     type=int,
    #     default=config.model.DEFAULT_CONTEXT_LENGTH,
    #     help=f"📏 Context length for the model (default: {config.model.DEFAULT_CONTEXT_LENGTH})",
    #     metavar="LENGTH"
    # )

    # Model stop command
    stop_command = model_subparsers.add_parser(
        "stop",
        help="🛑 Stop the running model server",
        description="Gracefully shutdown the currently running model server"
    )
    stop_command.add_argument(
        "--force",
        action="store_true",
        help="💥 Force kill processes immediately without graceful termination (use when normal stop fails)"
    )
    stop_command.add_argument(
        "--port",
        type=int,
        default=DEFAULT_CONFIG.network.DEFAULT_PORT,
        help=f"🌐 Port number for the server (default: {DEFAULT_CONFIG.network.DEFAULT_PORT})",
        metavar="PORT"
    )
    # Model download command
    download_command = model_subparsers.add_parser(
        "download",
        help="⬇️  Download model from IPFS",
        description="Download and extract model files from the decentralized network"
    )
    download_command.add_argument(
        "model_name",
        nargs='?',
        help="🏷️  Model name(s) - single: qwen3-1.7b or multi: qwen3-14b,qwen3-4b (first is main, others on-demand)"
    )
    download_command.add_argument(
        "--hash",
        type=str,
        help="🔗 Comma-separated Filecoin hashes (alternative to model names)",
        metavar="HASH"
    )
    download_command.add_argument(
        "--hf-repo",
        type=str,
        help="🤗 Hugging Face model repository",
        metavar="REPO"
    )
    download_command.add_argument(
        "--hf-file",
        type=str,
        help="🤗 Hugging Face model file",
        metavar="FILE"
    )
    download_command.add_argument(
        "--mmproj",
        type=str,
        help="🔍 Multimodal Projector File",
        metavar="MMProj"
    )
    download_command.add_argument(
        "--mmproj-url",
        type=str,
        help="🌐 URL to a multimodal projector file",
        metavar="URL"
    )
    download_command.add_argument(
        "--lora-config-path",
        type=str,
        help="🔍 Path to a lora config file",
        metavar="PATH"
    )
    # Model check command
    check_command = model_subparsers.add_parser(
        "check",
        help="🔍 Check if model is downloaded",
        description="Check if a model with the specified hash has been downloaded"
    )
    check_command.add_argument(
        "--hash",
        required=True,
        help="🔗 IPFS hash of the model to check",
        metavar="HASH"
    )

    # Model preserve command
    preserve_command = model_subparsers.add_parser(
        "preserve",
        help="💾 Preserve model to IPFS",
        description="Upload and preserve your model files to the decentralized network"
    )
    preserve_command.add_argument(
        "--task",
        type=str,
        default="chat",
        choices=["chat", "embed", "image-generation", "image-edit"],
        help="🎯 Model task type (default: chat)",
        metavar="TYPE"
    )
    preserve_command.add_argument(
        "--config-name",
        type=str,
        default=None,
        choices=["flux-dev", "flux-schnell"],
        help="🔍 Model config name (default: None), need for image-generation and image-edit models",
        metavar="CONFIG"
    )
    preserve_command.add_argument(
        "--gguf-folder",
        action="store_true",
        help="🔍 Indicate if this is a gguf folder include multiple files",
    )
    preserve_command.add_argument(
        "--lora",
        action="store_true",
        help="🔍 Indicate if this is a lora model (default: False)",
    )
    preserve_command.add_argument(
        "--folder-path",
        type=str,
        required=True,
        help="📂 Path to folder containing model files",
        metavar="PATH"
    )
    preserve_command.add_argument(
        "--zip-chunk-size",
        type=int,
        default=512,
        help="🗜️  Chunk size for splitting compressed files in MB (default: 512)",
        metavar="SIZE"
    )
    preserve_command.add_argument(
        "--threads",
        type=int,
        default=16,
        help="🧵 Number of compression threads (default: 16)",
        metavar="COUNT"
    )
    preserve_command.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="🔄 Maximum upload retry attempts (default: 5)",
        metavar="NUM"
    )
    preserve_command.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="🤗 Hugging Face model repository",
        metavar="REPO"
    )
    preserve_command.add_argument(
        "--hf-file",
        type=str,
        default=None,
        help="📄 Hugging Face model file",
        metavar="FILE"
    )
    preserve_command.add_argument(
        "--ram",
        type=float,
        default=None,
        help="🧠 Required RAM in GB for serving at 4096 context length",
        metavar="GB"
    )

    return parser.parse_known_args()

def handle_download(args) -> bool:
    """Handle model download with beautiful output"""
    # Determine the download source and validate inputs
    if args.hash:
        # Download by hash
        if args.hash not in HASH_TO_MODEL:
            print_error(f"Hash {args.hash} not found in HASH_TO_MODEL")
            sys.exit(1)
        model_name = HASH_TO_MODEL[args.hash]
        hf_data = FEATURED_MODELS[model_name]
        success, local_path = asyncio.run(download_model_async(hf_data, args.hash))
    elif args.model_name:
        if args.model_name in MODEL_TO_HASH:
            args.hash = MODEL_TO_HASH[args.model_name]
        if args.model_name not in FEATURED_MODELS:
            print_error(f"Model name {args.model_name} not found in FEATURED_MODELS")
            sys.exit(1)
        hf_data = FEATURED_MODELS[args.model_name]
        success, local_path = asyncio.run(download_model_async(hf_data, args.hash))
    else:
        # Download from Hugging Face
        hf_data = {
            "repo": args.hf_repo,
            "model": args.hf_file,
            "projector": args.mmproj,
        }
        success, local_path = asyncio.run(download_model_async(hf_data))
    
    # Handle download result
    if success:
        print_success(f"Model downloaded successfully to {local_path}")
        return True
    else:
        print_error("Download failed")
        sys.exit(1)

def handle_run(args):
    """Handle model loading and configuration based on provided arguments."""
    # Handle Hugging Face repository case separately
    if args.hf_repo:
        hf_data = {
            "repo": args.hf_repo,
            "model": args.hf_file,
            "projector": args.mmproj,
        }
        success, local_path = asyncio.run(download_model_async(hf_data))
        if not success:
            print_error(f"Failed to download model {args.hf_repo}")
            sys.exit(1)

        folder_name = os.path.basename(local_path)
        model_id = folder_name.replace("/", "_")
        if args.hf_file:
            local_path = os.path.join(local_path, args.hf_file)
            if os.path.exists(local_path):
                model_id = args.hf_file
            else:
                print_error(f"File {args.hf_file} not found in {local_path}")
                sys.exit(1)

        projector_path = None
        if args.mmproj:
            mmproj_path = os.path.join(local_path, args.mmproj)
            if os.path.exists(mmproj_path):
                projector_path = mmproj_path
                model_id = f"{model_id}_{args.mmproj}"

        config = {
            "model_id": model_id,
            "model": local_path,
            "context_length": args.context_length,
            "model_name": folder_name,
            "task": args.task,
            "on_demand": False,
            "is_lora": False,
            "projector": projector_path,
            "multimodal": bool(projector_path),
        }

        if args.config_name:
            config["config_name"] = args.config_name

        success =  manager.start([config], args.port, args.host)

        if not success:
            print_error(f"Failed to start model {model_id}")
            sys.exit(1)
        
        model_metadata_path = os.path.join(DEFAULT_MODEL_DIR, model_id + ".json")
        if not os.path.exists(model_metadata_path):
            with open(model_metadata_path, "w") as f:
                json.dump(config, f)
            print_success(f"Model metadata file {model_metadata_path} created")
        else:
            print_warning(f"Model metadata file {model_metadata_path} already exists")
        return success

    # Handle hash or model_name cases
    if args.hash:
        if args.hash not in HASH_TO_MODEL:
            print_error(f"Hash {args.hash} not found in HASH_TO_MODEL")
            sys.exit(1)
        model_name = HASH_TO_MODEL[args.hash]
    elif args.model_name:
        if args.model_name not in FEATURED_MODELS:
            print_error(f"Model name {args.model_name} not found in FEATURED_MODELS")
            sys.exit(1)
        model_name = args.model_name
        if model_name in MODEL_TO_HASH:
            args.hash = MODEL_TO_HASH[model_name]
    else:
        print_error("Either hash, model_name, or hf_repo must be provided")
        sys.exit(1)

    hf_data = FEATURED_MODELS[model_name]
    success, local_path = asyncio.run(download_model_async(hf_data, getattr(args, 'hash', None)))
    if not success:
        print_error(f"Failed to download model {model_name}")
        sys.exit(1)

    is_lora = False
    lora_config = None
    projector_path = None
    task = "chat"
    model_name_from_metadata = model_name

    # Fetch metadata if hash is available
    if hasattr(args, 'hash'):
        success, metadata = asyncio.run(fetch_model_metadata_async(args.hash))
        if not success:
            print_error(f"Failed to fetch model metadata for {args.hash}")
            sys.exit(1)
        is_lora = metadata.get("is_lora", False)
        task = metadata.get("task", "chat")
        model_name_from_metadata = metadata.get("folder_name", model_name)
    else:
        is_lora = hf_data.get("lora", False)
        task = hf_data.get("task", "chat")

    # Handle LoRA configuration
    if is_lora:
        metadata_path = os.path.join(local_path, "metadata.json")
        if not os.path.exists(metadata_path):
            print_error("LoRA model found but metadata.json is missing")
            sys.exit(1)
        with open(metadata_path, "r") as f:
            lora_metadata = json.load(f)
        lora_paths = lora_metadata.get("lora_paths", [])
        lora_scales = lora_metadata.get("lora_scales", [])
        base_model_hash = lora_metadata.get("base_model")
        if base_model_hash not in HASH_TO_MODEL:
            print_error(f"Base model hash {base_model_hash} not found")
            sys.exit(1)
        base_model_name = HASH_TO_MODEL[base_model_hash]
        base_model_hf_data = FEATURED_MODELS[base_model_name]
        success, base_model_local_path = asyncio.run(download_model_async(base_model_hf_data, base_model_hash))
        if not success:
            print_error(f"Failed to download base model {base_model_hash}")
        local_path = base_model_local_path
        lora_config = dict(zip(lora_paths, lora_scales))

    # Determine projector path
    projector_candidates = [
        f"{local_path}-projector",
        os.path.join(local_path, hf_data.get("projector", "")) if "projector" in hf_data else None
    ]
    for candidate in projector_candidates:
        if candidate and os.path.exists(candidate):
            projector_path = candidate
            break

    model_id = args.hash if hasattr(args, 'hash') else model_name
    # Build configuration
    config = {
        "model_id": model_id,
        "hash": getattr(args, 'hash', None),
        "model": local_path,
        "context_length": args.context_length,
        "model_name": model_name_from_metadata,
        "task": task,
        "projector": projector_path,
        "multimodal": bool(projector_path),
        "on_demand": False,
        "is_lora": is_lora,
        "lora_config": lora_config,
    }

    success = manager.start([config], args.port, args.host)

    if not success:
        print_error(f"Failed to start model {model_id}")
        sys.exit(1)

    model_metadata_path = os.path.join(DEFAULT_MODEL_DIR, model_id + ".json")
    if not os.path.exists(model_metadata_path):
        with open(model_metadata_path, "w") as f:
            json.dump(config, f)
        print_success(f"Model metadata file {model_metadata_path} created")
    else:
        print_warning(f"Model metadata file {model_metadata_path} already exists")
    
    return success


def handle_serve(args):
    """Handle model serve command - run all downloaded models with specified main model"""
    print_info("Discovering downloaded models in llms-storage...")

    # Get all downloaded models
    downloaded_models = get_all_downloaded_models()

    if not downloaded_models:
        print_error("No downloaded models found in llms-storage directory")
        print_info("Use 'eai model download --hash <hash>' to download models first")
        sys.exit(1)

    print_success(f"Found {len(downloaded_models)} downloaded model(s)")

    # Handle main hash selection
    main_hash = args.main_hash
    if main_hash is None:
        # Randomly select a model as main
        main_hash = random.choice(downloaded_models)
        print_info(f"No main hash specified, randomly selected: {main_hash}")
    else:
        # Validate that main hash exists among downloaded models
        if main_hash not in downloaded_models:
            print_error(f"Main model hash '{main_hash}' not found in downloaded models")
            print_warning("Available downloaded models:")
            for i, model_hash in enumerate(downloaded_models, 1):
                print(f"  {i}. {model_hash}")
            sys.exit(1)
        print_success(f"Using specified main model hash: {main_hash}")

    # Prepare the hash string for multi-model startup
    # Put main hash first, then all others (excluding main hash to avoid duplication)
    other_hashes = [h for h in downloaded_models if h != main_hash]
    all_hashes = [main_hash] + other_hashes
    model_hashes_str = ','.join(all_hashes)

    if len(all_hashes) > 1:
        print_info(f"Multi-model setup:")
        print_info(f"  Main model (loaded immediately): {main_hash}")
        print_info(f"  On-demand models ({len(other_hashes)}): {', '.join(other_hashes[:3])}" +
                  ("..." if len(other_hashes) > 3 else ""))
    else:
        print_info(f"Single model: {main_hash}")

    print_info(f"Starting model server...")
    print_info(f"Host: {args.host}, Port: {args.port}, Context: {args.context_length}")

    if not manager.start(model_hashes_str, args.port, args.host, args.context_length):
        print_error("Failed to start model server")
        sys.exit(1)
    else:
        print_success(f"Model server started successfully on {args.host}:{args.port}")
        print_info(f"Serving {len(all_hashes)} model(s) with main model: {main_hash}")

def handle_stop(args):
    """Handle model stop with beautiful output"""
    if not manager.stop():
        print_error("Failed to stop model server or no server running")
    else:
        print_success("Model server stopped successfully")

def handle_preserve(args):
    """Handle model preservation with beautiful output"""
    print_info(f"Starting preservation of: {args.folder_path}")
    print_info(f"Task: {args.task}, Threads: {args.threads}, Chunk size: {args.zip_chunk_size}MB")

    kwargs = {
        "task": args.task,
        "ram": args.ram,
        "config_name": args.config_name,
        "hf_repo": args.hf_repo,
        "hf_file": args.hf_file,
        "lora": args.lora,
        "gguf_folder": args.gguf_folder,
    }

    try:
        upload_folder_to_lighthouse(args.folder_path, args.zip_chunk_size, args.max_retries, args.threads, **kwargs)
        print_success("Model preserved successfully to IPFS!")
    except Exception as e:
        print_error(f"Preservation failed: {str(e)}")
        sys.exit(1)

def handle_check(args):
    """Handle model check with beautiful output"""
    print_info(f"Checking if model is downloaded for hash: {args.hash}")
    try:
        local_path = DEFAULT_MODEL_DIR / f"{args.hash}{POSTFIX_MODEL_PATH}"
        is_downloaded = local_path.exists()

        if is_downloaded:
            # For LoRA models, we need to do additional validation
            if local_path.is_dir():
                # This is likely a LoRA model - check if it has valid metadata and base model
                metadata_path = local_path / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            lora_metadata = json.load(f)

                        # Check if base model is available
                        base_model_hash = lora_metadata.get("base_model")
                        if base_model_hash:
                            base_model_path = DEFAULT_MODEL_DIR / f"{base_model_hash}{POSTFIX_MODEL_PATH}"
                            if not base_model_path.exists():
                                print_warning(f"LoRA model found but base model missing: {base_model_hash}")
                                print_info("False")
                                return

                        # Check if LoRA files exist
                        lora_paths = lora_metadata.get("lora_paths", [])
                        for lora_path in lora_paths:
                            if not os.path.isabs(lora_path):
                                lora_path = os.path.join(local_path, lora_path)
                            if not os.path.exists(lora_path):
                                print_warning(f"LoRA model found but LoRA file missing: {lora_path}")
                                print_info("False")
                                return

                        print_success("True")
                    except (json.JSONDecodeError, KeyError, Exception) as e:
                        print_warning(f"LoRA model found but metadata is invalid: {str(e)}")
                        print_info("False")
                else:
                    print_warning("LoRA model directory found but metadata.json is missing")
                    print_info("False")
            else:
                # Regular model file
                print_success("True")
        else:
            print_info("False")
    except Exception as e:
        print_error(f"Check failed: {str(e)}")
        print_info("False")
        sys.exit(1)

def main():
    """Main CLI entry point with enhanced error handling"""
    # Show banner
    print_banner()

    known_args, unknown_args = parse_args()

    # Handle unknown arguments
    if unknown_args:
        for arg in unknown_args:
            print_error(f'Unknown command or argument: {arg}')
        print_info("Use --help for available commands and options")
        sys.exit(2)

    # Handle commands
    if known_args.command == "model":
        if known_args.model_command == "run":
            handle_run(known_args)
        # elif known_args.model_command == "serve":
        #     handle_serve(known_args)
        elif known_args.model_command == "stop":
            handle_stop(known_args)
        elif known_args.model_command == "download":
            handle_download(known_args)
        # elif known_args.model_command == "preserve":
        #     handle_preserve(known_args)
        # elif known_args.model_command == "check":
        #     handle_check(known_args)
        else:
            print_error(f"Unknown model command: {known_args.model_command}")
            print_info("Available model commands: run, serve, stop, download, status, preserve, check")
            sys.exit(2)
    else:
        print_error(f"Unknown command: {known_args.command}")
        print_info("Available commands: model")
        print_info("Use --help for more information")
        sys.exit(2)


if __name__ == "__main__":
    main()