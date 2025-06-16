# Deploy Your Local Large Language Model to CryptoAgents

This guide will help you deploy your local AI models to the CryptoAgents platform using decentralized infrastructure. Whether you're a developer or AI enthusiast, you'll learn how to run your models securely and efficiently.

## 🌟 Key Features

- **Decentralized Deployment**: Run your models on a distributed network
- **Secure Storage**: Store models using IPFS/Filecoin
- **OpenAI Compatibility**: Use familiar API endpoints
- **Multi-Model Support**: Works with both text and vision models

## 🛠️ Getting Started

### Installation

#### For macOS:
```bash
bash mac.sh
```
> **Note**: You'll need `llama.cpp.rb` in the same directory as `mac.sh`

### Setting Up Your Environment

1. Activate the virtual environment:
```bash
source local_ai/bin/activate
```
> **Remember**: Activate this environment each time you use the `local-ai` tools

## 🚀 Running Models

### Available Pre-uploaded Models

We've prepared several models for you to test with:

#### 🔤 Qwen3 Series
[Learn more about Qwen3](https://qwenlm.github.io/blog/qwen3/)

**Qwen3-4B-Q8**
- Size: 4.28 GB
- RAM Required: 9.5 GB
- CID: `bafkreiekokvzioogj5hoxgxlorqvbw2ed3w4mwieium5old5jq3iubixza`
- Command: `local-ai start --hash bafkreiekokvzioogj5hoxgxlorqvbw2ed3w4mwieium5old5jq3iubixza`

**Qwen3-8B-Q6**
- Size: 6.21 GB
- RAM Required: 12 GB
- CID: `bafkreid5z4lddvv4qbgdlz2nqo6eumxwetwmkpesrumisx72k3ahq73zpy`
- Command: `local-ai start --hash bafkreid5z4lddvv4qbgdlz2nqo6eumxwetwmkpesrumisx72k3ahq73zpy`

**Qwen3-14B-Q8**
- Size: 15.7 GB
- RAM Required: 19.5 GB
- CID: `bafkreiclwlxc56ppozipczuwkmgnlrxrerrvaubc5uhvfs3g2hp3lftrwm`
- Command: `local-ai start --hash bafkreiclwlxc56ppozipczuwkmgnlrxrerrvaubc5uhvfs3g2hp3lftrwm`

#### 👁️ Gemma3 Series (Vision Support)
[Learn more about Gemma3](https://deepmind.google/models/gemma/gemma-3/)

**Gemma-4B-IT-Q4**
- Size: 3.16 GB
- RAM Required: 7.9 GB
- CID: `bafkreiaevddz5ssjnbkmdrl6dzw5sugwirzi7wput7z2ttcwnvj2wiiw5q`
- Command: `local-ai start --hash bafkreiaevddz5ssjnbkmdrl6dzw5sugwirzi7wput7z2ttcwnvj2wiiw5q`

**Gemma-12B-IT-Q4**
- Size: 8.07 GB
- RAM Required: 21.46 GB
- CID: `bafkreic2bkjuu3fvdoxnvusdt4in6fa6lubzhtjtmcp2zvokvfjpyndakq`
- Command: `local-ai start --hash bafkreic2bkjuu3fvdoxnvusdt4in6fa6lubzhtjtmcp2zvokvfjpyndakq`

**Gemma-27B-IT-Q4**
- Size: 17.2 GB
- RAM Required: 38.0 GB
- CID: `bafkreihi2cbsgja5dwa5nsuixicx2x3gbcnh7gsocxbmjxegtewoq2syve`
- Command: `local-ai start --hash bafkreihi2cbsgja5dwa5nsuixicx2x3gbcnh7gsocxbmjxegtewoq2syve`

## 💻 Using the API

### Text Chat Example
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

### Vision Chat Example (Gemma models only)
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

## 📚 Additional Information

### Model Format
- We support GGUF files compatible with llama.cpp
- Convert other formats using [llama.cpp conversion tools](https://github.com/ggerganov/llama.cpp)

### Performance Tips
- Choose quantization levels (Q4, Q6, Q8) based on your hardware capabilities
- Higher quantization (Q8) offers better quality but requires more resources
- Lower quantization (Q4) is more efficient but may affect model performance

### Security
- All models are stored on IPFS with content addressing
- This ensures model integrity and secure distribution

## 🆘 Need Help?

- Visit our website: [eternalai.org](https://eternalai.org)
- Join our community: [Discord](https://discord.gg/YphRKtSFqS)
- Check our documentation for detailed guides and tutorials