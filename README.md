# Deploy Your Local Large Language Model to CryptoAgents

This guide will help you deploy your local AI models to the CryptoAgents platform using decentralized infrastructure. Whether you're a developer or AI enthusiast, you'll learn how to run your models securely and efficiently.

## 🌟 Key Features

- **Decentralized Deployment**: Run your models on a distributed network
- **Secure Storage**: Store models using IPFS/Filecoin
- **OpenAI Compatibility**: Use familiar API endpoints
- **Multi-Model Support**: Works with both text and vision models
- **Parallel Processing**: Efficient model compression and upload
- **Automatic Retries**: Robust error handling for network issues
- **Metadata Management**: Comprehensive model information tracking

## 📋 Before You Start

Make sure you have:
1. A model in `gguf` format (compatible with `llama.cpp`)
2. A [Lighthouse](https://lighthouse.storage/) account and API key

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

### Verification

Verify your installation:
```bash
local-ai --version
```

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

## Advanced Usage

### Uploading Custom Models

You can use `local-ai upload` to upload your own `gguf` models downloaded from [Huggingface](https://huggingface.co/) for deploying to the CryptoAgents platform.

#### Model Preparation

1. **Download the model**: Download the `.gguf` file from Huggingface (e.g., [`Qwen3-8B-Q8_0.gguf`](https://huggingface.co/Qwen/Qwen3-8B-GGUF/blob/main/Qwen3-8B-Q8_0.gguf))
2. **Create a folder**: Create a folder with a descriptive name (e.g., `qwen3-8b-q8`)
3. **Rename the file**: Place the model file in the folder and rename it to match the folder name **without** the `.gguf` extension

**Example Structure:**
```
qwen3-8b-q8/
└── qwen3-8b-q8  # Note: no .gguf extension
```

#### Estimating RAM Requirements

Use the [GGUF parser](https://www.npmjs.com/package/@huggingface/gguf) to estimate RAM usage:

```bash
npx @huggingface/gguf qwen3-8b-q8/qwen3-8b-q8 --context 32768
```

#### Upload Command

**Basic Upload:**
```bash
export LIGHTHOUSE_API_KEY=your_api_key
local-ai upload --folder-name qwen3-8b-q8
```

**Advanced Upload with Metadata:**
```bash
export LIGHTHOUSE_API_KEY=your_api_key
local-ai upload \
  --folder-name qwen3-8b-q8 \
  --ram 12 \
  --hf-repo Qwen/Qwen3-8B-GGUF \
  --hf-file Qwen3-8B-Q8_0.gguf \
  --zip-chunk-size 512 \
  --threads 16 \
  --max-retries 20
```

#### Upload Command Options

| Option | Description | Default | Required |
|--------|-------------|---------|----------|
| `--folder-name` | Folder containing the model files | - | ✅ |
| `--ram` | RAM usage in GB at 32768 context length | - | ❌ |
| `--hf-repo` | Hugging Face repository (e.g., `Qwen/Qwen3-8B-GGUF`) | - | ❌ |
| `--hf-file` | Original Hugging Face filename | - | ❌ |
| `--zip-chunk-size` | Compression chunk size in MB | 512 | ❌ |
| `--threads` | Number of compression threads | 16 | ❌ |
| `--max-retries` | Maximum upload retry attempts | 20 | ❌ |

#### Upload Process Details

The upload process involves several steps:

1. **Compression**: The model folder is compressed using `tar` and `pigz` for optimal compression
2. **Chunking**: Large files are split into chunks (default: 512MB) for reliable uploads
3. **Parallel Upload**: Multiple chunks are uploaded simultaneously for faster transfer
4. **Retry Logic**: Failed uploads are automatically retried up to 20 times
5. **Metadata Generation**: A metadata file is created with upload information and model details
6. **IPFS Storage**: All files are stored on IPFS via Lighthouse.storage

#### Example Output

After a successful upload, you'll receive:
- **CID (Content Identifier)**: Used to reference your uploaded model
- **Metadata file**: Contains detailed information about the upload
- **Upload statistics**: File sizes, upload speeds, and timing information

#### Troubleshooting Upload Issues

**Common Issues:**
- **Missing API Key**: Ensure `LIGHTHOUSE_API_KEY` is set in your environment
- **Network Issues**: The system will automatically retry failed uploads

## 📚 Additional Information

### Model Format
- We support GGUF files compatible with llama.cpp
- Convert other formats using [llama.cpp conversion tools](https://github.com/ggerganov/llama.cpp)

### Performance Tips
- Choose quantization levels (Q4, Q6, Q8) based on your hardware capabilities
- Higher quantization (Q8) offers better quality but requires more resources
- Lower quantization (Q4) is more efficient but may affect model performance
- Monitor system resources during model operation
- Use appropriate context lengths for your use case

### Security
- All models are stored on IPFS with content addressing
- This ensures model integrity and secure distribution
- API keys are stored securely in environment variables
- Models are verified before deployment

### Best Practices
1. **Model Selection**
   - Choose models based on your hardware capabilities
   - Consider quantization levels for optimal performance
   - Test models locally before deployment

2. **Resource Management**
   - Monitor RAM usage during model operation
   - Adjust context length based on available memory
   - Use appropriate batch sizes for your use case

3. **API Usage**
   - Implement proper error handling
   - Use appropriate timeouts for requests
   - Cache responses when possible
   - Monitor API usage and performance

4. **Deployment**
   - Test models thoroughly before production use
   - Keep track of model versions and CIDs
   - Document model configurations and requirements
   - Regular backups of model metadata

## 🆘 Need Help?

- Visit our website: [eternalai.org](https://eternalai.org)
- Join our community: [Discord](https://discord.gg/YphRKtSFqS)
- Check our documentation for detailed guides and tutorials
- Report issues on our GitHub repository
- Contact support for enterprise assistance