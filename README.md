# Deploy Your Local Large Language Model to CryptoAgents

This comprehensive guide explains how to deploy your local large language model to the CryptoAgents platform using decentralized storage and compute infrastructure.

## 🚀 Overview

CryptoAgents enables you to:
- Deploy local models to a decentralized network
- Use IPFS/Filecoin for distributed model storage
- Run models with OpenAI-compatible API endpoints
- Support both text and vision models

## 📋 Prerequisites

Before you begin, ensure you have:

- **Model Format**: Your local model must be in `gguf` format (compatible with `llama.cpp`)
- **Lighthouse Account**: Create an account at [Lighthouse](https://lighthouse.storage/) and obtain your API key

## 🛠️ Installation

### Step 1: Install the Local AI Tool

The installation process varies by operating system:

#### For macOS Users:
```bash
bash mac.sh
```

To successfully run `mac.sh`, you need the file `llama.cpp.rb` in the same directory as `mac.sh`.

## Step 2: Activate the Virtual Environment

```bash
source local_ai/bin/activate
```

**Note**: Remember to activate this environment every time you want to use the `local-ai` tools.

## 📤 Uploading Your Model

### Step 3: Prepare Your Model Files

The `local-ai` tool requires a specific folder structure:

1. **Create a folder** with the same name as your model (without the file extension)
2. **Move your model file** inside this folder 
3. **Rename the model file** to match the folder name (remove the `.gguf` extension)

**Example:**
```bash
# Original structure
llama3-8b.gguf

# Required structure after preparation
llama3-8b/
└── llama3-8b  # Renamed from llama3-8b.gguf
```

**Quick commands to set this up:**
```bash
# Replace 'llama3-8b.gguf' with your actual model filename
MODEL_NAME="llama3-8b"
mkdir "${MODEL_NAME}"
mv "${MODEL_NAME}.gguf" "${MODEL_NAME}/${MODEL_NAME}"
```

### Step 4: Upload to Lighthouse

#### Basic Upload
```bash
local-ai upload --folder-name llama3-8b
```

#### Upload with Metadata (Recommended)
You can specify additional metadata to help users understand your model's requirements:

```bash
local-ai upload --folder-name llama3-8b \
  --ram "9.5 GB" \
  --hf-repo "meta-llama/Llama-3-8B" \
  --hf-file "llama3-8b.gguf"
```

**Metadata Parameters:**

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--ram` | Required RAM for 32k context window | `"9.5 GB"` |
| `--hf-repo` | HuggingFace repository ID | `"meta-llama/Llama-3-8B"` |
| `--hf-file` | Original filename in HuggingFace repo | `"llama3-8b.gguf"` |

After upload completion, metadata will be saved in `{model_name}_metadata.json` containing the **Filecoin hash (IPFS CID)** needed for deployment.

## 🖥️ Local Testing

### Step 5: Run Your Model Locally

After uploading, find your `cid` in the generated metadata file:

```bash
# Check your metadata file
cat llama3-8b_metadata.json

# Start your model using the CID
local-ai start --hash ${cid}
```

The model will start on **port 8080** by default.

### Pre-uploaded Models

For quick testing, we have several models already available:

#### 🔤 **Qwen3 Series** ([Learn more](https://qwenlm.github.io/blog/qwen3/))

| Model | Size | RAM Required | CID | Start Command |
|-------|------|--------------|-----|---------------|
| **Qwen3-4B-Q8** | 4.3GB | 9.5 GB | `bafkreiekokvzioogj5hoxgxlorqvbw2ed3w4mwieium5old5jq3iubixza` | `local-ai start --hash bafkreiekokvzioogj5hoxgxlorqvbw2ed3w4mwieium5old5jq3iubixza` |
| **Qwen3-8B-Q6** | 6.8GB | 12 GB | `bafkreid5z4lddvv4qbgdlz2nqo6eumxwetwmkpesrumisx72k3ahq73zpy` | `local-ai start --hash bafkreid5z4lddvv4qbgdlz2nqo6eumxwetwmkpesrumisx72k3ahq73zpy` |
| **Qwen3-14B-Q8** | 15.8GB | 19.5 GB | `bafkreiclwlxc56ppozipczuwkmgnlrxrerrvaubc5uhvfs3g2hp3lftrwm` | `local-ai start --hash bafkreiclwlxc56ppozipczuwkmgnlrxrerrvaubc5uhvfs3g2hp3lftrwm` |

#### 👁️ **Gemma3 Series** ([Learn more](https://deepmind.google/models/gemma/gemma-3/)) - **Vision Support**

| Model | Size | RAM Required | CID | Start Command |
|-------|------|--------------|-----|---------------|
| **Gemma-4B-IT-Q4** | 2.8GB | 7.9 GB | `bafkreiaevddz5ssjnbkmdrl6dzw5sugwirzi7wput7z2ttcwnvj2wiiw5q` | `local-ai start --hash bafkreiaevddz5ssjnbkmdrl6dzw5sugwirzi7wput7z2ttcwnvj2wiiw5q` |
| **Gemma-12B-IT-Q4** | 8.9GB | 21.46 GB | `bafkreic2bkjuu3fvdoxnvusdt4in6fa6lubzhtjtmcp2zvokvfjpyndakq` | `local-ai start --hash bafkreic2bkjuu3fvdoxnvusdt4in6fa6lubzhtjtmcp2zvokvfjpyndakq` |
| **Gemma-27B-IT-Q4** | 18.2GB | 38.0 GB | `bafkreihi2cbsgja5dwa5nsuixicx2x3gbcnh7gsocxbmjxegtewoq2syve` | `local-ai start --hash bafkreihi2cbsgja5dwa5nsuixicx2x3gbcnh7gsocxbmjxegtewoq2syve` |

### Step 6: Test Your Model

#### 💬 Text Chat Completion
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [
        {"role": "user", "content": "Hello! Can you help me write a Python function?"}
    ],
    "temperature": 0.7,
    "max_tokens": 4096
}'
```

#### 👁️ Vision Chat Completion (Gemma models only)
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [
       {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What do you see in this image? Please describe it in detail."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/your-image.jpg"
                }
            }
        ]
       }
    ],
    "temperature": 0.7,
    "max_tokens": 4096
}'
```

## 🌐 Deployment to CryptoAgents Platform

### Step 7: Deploy to Production

1. **Visit the Creation Portal**: Go to [CryptoAgents Create Agent](https://staging.eternalai.org/for-developers/create?tab=5)

2. **Configure Your Agent** with the following settings:

| Field | Value | Description |
|-------|-------|-------------|
| **Agent type** | `Large Language Model` | Type of AI agent |
| **Category** | `Model` | Agent category |
| **Network** | `Base Sepolia` | Blockchain network |
| **Agent avatar** | Upload image | Visual representation |
| **Agent name** | Your model's name | Unique identifier |
| **Display name** | Friendly name | User-facing name |
| **Short description** | Brief summary | One-line description |
| **Description** | Detailed info | Full model description |
| **Model** | `ipfs://[YOUR_CID]` | IPFS hash from metadata file |
| **Dependent agents** | Leave blank | No dependencies |
| **Required free RAM** | From metadata | RAM needed (32k context) |
| **Required free Disk** | Model size × 2 | Storage needed (includes decompression) |

**Example Model Field:**
```
ipfs://bafkreiekokvzioogj5hoxgxlorqvbw2ed3w4mwieium5old5jq3iubixza
```

## 🔧 Troubleshooting

### Common Issues and Solutions

| Issue | Possible Causes | Solutions |
|-------|----------------|-----------|
| **Upload fails** | Invalid API key, network issues | • Verify Lighthouse API key<br>• Check internet connection<br>• Try uploading smaller chunks |
| **Model won't start** | Insufficient RAM, corrupted files | • Check system RAM vs requirements<br>• Re-download/re-upload model<br>• Verify CID accuracy |
| **API errors** | Wrong port, model not loaded | • Confirm model is running on port 8080<br>• Wait for model to fully load<br>• Check logs for errors |
| **Folder structure error** | Incorrect naming convention | • Ensure folder name matches model name<br>• Remove `.gguf` extension from file inside folder |

### Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/eternalai-org/local-ai/issues)
- **Documentation**: Check the `/docs` folder for additional guides
- **Community**: Join our Discord for community support

### System Requirements Check

Before deployment, verify your system meets the requirements:

```bash
# Check available RAM
free -h

# Check available disk space
df -h

# Verify Python version
python3 --version

# Test virtual environment
source local_ai/bin/activate && local-ai --version
```

## 📚 Additional Resources

- **Supported Model Formats**: GGUF files compatible with llama.cpp
- **Model Conversion**: Use [llama.cpp conversion tools](https://github.com/ggerganov/llama.cpp) for other formats
- **Performance Optimization**: Consider quantization levels (Q4, Q6, Q8) based on your hardware
- **Security**: Models are stored on IPFS with content addressing for integrity

---

**Need help?** Contact us via [eternalai.org](https://eternalai.org)