# Deploy Your Local Large Language Model to CryptoAgents

This guide explains how to deploy your local large language model to the CryptoAgents platform.

## Prerequisites

- Your local model must be in `gguf` format (compatible with `llama.cpp`)
- You need a Lighthouse account with an API key for uploading the model

## Step 1: Install the Local AI Tool

Install the `local-ai` tool required for uploading models to Lighthouse:

```bash
bash mac.sh
```

## Step 2: Activate the Virtual Environment

```bash
source local_ai/bin/activate
```

## Step 3: Upload Your Model to Lighthouse

### Prepare Your Model Files

The `local-ai` tool requires a specific folder structure for your model:

1. Create a folder with the same name as your model (without the file extension)
2. Place your model file inside this folder and rename it to match the folder name

Example:
- Original model file: `llama3-8b.gguf`
- Create folder: `llama3-8b`
- Place the model file inside the folder and rename it to `llama3-8b`

After doing this, your folder structure should look like this:

```bash
llama3-8b/
└── llama3-8b  # This was renamed from llama3-8b.gguf
```

### Upload the Model

```bash
local-ai upload --folder-name llama3-8b
```

### Model Metadata Options

You can specify additional metadata for your model:

| Parameter | Description |
|-----------|-------------|
| `ram` | Required RAM amount for 32k context window |
| `hf-repo` | HuggingFace repository ID of the model |
| `hf-file` | File name of the model in HuggingFace repo |

After uploading completes, the metadata will be saved in a file named `llama3-8b_metadata.json`. This file contains the Filecoin hash (IPFS CID) needed for deployment.

## Step 4: Try running your model locally

### Start the model locally

After uploading the model, you can find your `cid` in the `llama3-8b_metadata.json` file.

To run the model locally:
```bash
local-ai start --hash ${cid}
```

For instance, we have already uploaded several models to lighthouse such as:
- [Qwen3 series](https://qwenlm.github.io/blog/qwen3/)
    - Qwen3-4B-Q8-GGUF: 
        - cid: `bafkreiekokvzioogj5hoxgxlorqvbw2ed3w4mwieium5old5jq3iubixza`
        - ram: 9.5 GB RAM (for 32k context window)
        - run:
            ```bash
            local-ai start --hash bafkreiekokvzioogj5hoxgxlorqvbw2ed3w4mwieium5old5jq3iubixza
            ```
    - Qwen3-8B-Q6-GGUF:
        - cid: `bafkreid5z4lddvv4qbgdlz2nqo6eumxwetwmkpesrumisx72k3ahq73zpy`
        - ram: 12 GB RAM (for 32k context window)
        - run:
            ```bash
            local-ai start --hash bafkreid5z4lddvv4qbgdlz2nqo6eumxwetwmkpesrumisx72k3ahq73zpy
            ```
    - Qwen3-14B-Q8-GGUF:
        - cid: `bafkreiclwlxc56ppozipczuwkmgnlrxrerrvaubc5uhvfs3g2hp3lftrwm`
        - ram: 19.5 GB RAM (for 32k context window)
        - run:
            ```bash
            local-ai start --hash bafkreiclwlxc56ppozipczuwkmgnlrxrerrvaubc5uhvfs3g2hp3lftrwm
            ```
- [Gemma3 series](https://deepmind.google/models/gemma/gemma-3/) (has vision support)
    - Gemma-4B-IT-QAT-Q4-GGUF: 
        - cid: `bafkreiaevddz5ssjnbkmdrl6dzw5sugwirzi7wput7z2ttcwnvj2wiiw5q`
        - ram: 7.9 GB RAM (for 32k context window)
        - run:
            ```bash
            local-ai start --hash bafkreiaevddz5ssjnbkmdrl6dzw5sugwirzi7wput7z2ttcwnvj2wiiw5q
            ```
    - Gemma-3-12B-IT-QAT-Q4-GGUF:
        - cid: `bafkreic2bkjuu3fvdoxnvusdt4in6fa6lubzhtjtmcp2zvokvfjpyndakq`
        - ram: 21.46 GB RAM (for 32k context window)
        - run:
            ```bash
            local-ai start --hash bafkreic2bkjuu3fvdoxnvusdt4in6fa6lubzhtjtmcp2zvokvfjpyndakq
            ```
    - Gemma-3-27B-IT-QAT-Q4-GGUF:
        - cid: `bafkreihi2cbsgja5dwa5nsuixicx2x3gbcnh7gsocxbmjxegtewoq2syve`
        - ram: 38.0 GB RAM (for 32k context window)
        - run:
            ```bash
            local-ai start --hash bafkreihi2cbsgja5dwa5nsuixicx2x3gbcnh7gsocxbmjxegtewoq2syve
            ```
By default, the model will run on port 8080.

### Test the model

1. Simple chat completion:
```bash
curl -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "local-model",
    "messages": [
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 4096
}'
```
2. Vision chat completion:
```bash
curl -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "local-model",
    "messages": [
       {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What do you see in this image?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.jpg"
                }
            }
        ]
       }
    ],
    "temperature": 0.7,
    "max_tokens": 4096
}'
```

## Step 5: Deploy Your Model to Vibe

1. Go to the [Vibe Create Agent Website](https://staging.eternalai.org/for-developers/create?tab=5)
2. Create a new local model agent using the following information:

### Agent Configuration

| Field | Description |
|-------|-------------|
| Agent type | Large Language Model |
| Category | Model |
| Network | Base Sepolia |
| Agent avatar | Your model's avatar image |
| Agent name | Your model's name |
| Display name | Your model's display name |
| Short description | Brief description of your model |
| Description | Detailed description of your model |
| Model | `ipfs://[YOUR_FILECOIN_HASH]` (e.g., `ipfs://bafkreibl7efovdbo5vb3u6pcga2e2s7bqnyf7lrgaliaeldlaqtvgjmd4q`) |
| Dependent agents | Leave blank |
| Required free RAM | Estimated RAM needed for 32k context window |
| Required free Disk | Your model size in GB (double the actual size to account for decompression) |

## Troubleshooting

If you encounter issues during the upload or deployment process, check:
- Your API key is valid and has sufficient permissions
- The model file is in the correct format
- The folder structure follows the naming convention
- You have sufficient disk space and RAM for the model 