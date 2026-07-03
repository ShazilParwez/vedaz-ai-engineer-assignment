# Hosting Vedaz Qwen2.5 with vLLM

## 1. Overview

**What is vLLM?**
vLLM is a blazing-fast, open-source library for large language model (LLM) inference and serving. It employs a novel attention algorithm called PagedAttention, which intelligently manages attention keys and values (KV cache) in a non-contiguous, highly efficient manner.

**Why use it for inference?**
- **Throughput:** State-of-the-art serving throughput, frequently outperforming Hugging Face Transformers by 10x-20x in concurrent request scenarios.
- **Memory Efficiency:** PagedAttention virtually eliminates memory fragmentation.
- **OpenAI Compatibility:** Natively provides an OpenAI-compatible API server, allowing drop-in replacements for existing GPT-3.5/4 applications.
- **Dynamic LoRA Support:** Allows serving a base model alongside multiple LoRA adapters simultaneously without merging weights.

---

## 2. Prerequisites

- **OS:** Ubuntu 22.04 LTS
- **Hardware:** NVIDIA GPU (Ampere/Ada architecture recommended, e.g., RTX 3090, 4090, A10G, A100)
- **Drivers:** CUDA Toolkit 12.1 or later installed
- **Software:** Python 3.10+

---

## 3. Install Dependencies

Update your system packages and install the necessary base dependencies:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git curl wget
```

---

## 4. Clone the Project

Clone the Vedaz repository onto your VPS and enter the directory:

```bash
git clone https://github.com/your-org/vedaz_ai_engineer.git
cd vedaz_ai_engineer
```

*(Note: If you generated the LoRA outputs locally, securely transfer the `./outputs` directory to this server via `scp` or `rsync`.)*

---

## 5. Install vLLM

It is highly recommended to use a virtual Python environment to prevent dependency conflicts on your server.

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install vllm openai huggingface_hub
```

---

## 6. Download the Base Qwen2.5 Model

While vLLM can download the model automatically upon first run, downloading it explicitly in advance ensures network stability and caches the weights in `~/.cache/huggingface`.

```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct
```

---

## 7. Load the LoRA Adapter

vLLM supports serving LoRA adapters dynamically over a base model. You must explicitly enable LoRA routing when starting the server. Assuming your fine-tuned LoRA weights are stored in the `./outputs` directory:

- **Base Model:** `Qwen/Qwen2.5-3B-Instruct`
- **LoRA Name:** `vedaz-astrologer`
- **LoRA Path:** `./outputs`

---

## 8. Launch the OpenAI-Compatible Server

Start the vLLM OpenAI-compatible API server. This exposes a REST endpoint at `http://localhost:8000`.

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --enable-lora \
    --lora-modules vedaz-astrologer=./outputs \
    --max-lora-rank 16 \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 \
    --port 8000
```

---

## 9. Example `curl` Request

Because vLLM mimics the OpenAI API schema, you can query it precisely as you would GPT-4. Notice that we pass `vedaz-astrologer` as the `model` parameter to dynamically target our specific LoRA adapter.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vedaz-astrologer",
    "messages": [
      {"role": "system", "content": "You are Vedaz, a compassionate AI Astrologer."},
      {"role": "user", "content": "I am facing delays in my career. What does astrology say?"}
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

---

## 10. Example Python Client Request

You can use the official `openai` Python SDK by simply pointing the `base_url` to your local vLLM instance.

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY", # vLLM does not require an API key by default
    base_url="http://localhost:8000/v1",
)

response = client.chat.completions.create(
    model="vedaz-astrologer", # Triggers the LoRA adapter inference
    messages=[
        {"role": "system", "content": "You are Vedaz, a compassionate AI Astrologer."},
        {"role": "user", "content": "I am facing delays in my career. What does astrology say?"}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)
```

---

## 11. Performance Considerations

- **GPU Memory (`--gpu-memory-utilization`)**: By default, vLLM allocates 90% of your GPU memory for the KV cache to maximize throughput. If you are running other processes on the same GPU, lower this value (e.g., `--gpu-memory-utilization 0.50`).
- **Tensor Parallelism (`--tensor-parallel-size`)**: If you have multiple GPUs (e.g., two RTX 3090s), you can split the model across them for faster inference and larger memory pools by adding `--tensor-parallel-size 2`.
- **Batch Inference**: vLLM's PagedAttention automatically handles continuous batching. You do not need to configure batch sizes manually; the server natively maximizes concurrent request throughput under the hood.

---

## 12. Production Recommendations

### systemd Service
To ensure the server restarts automatically upon crashes or server reboots, create a `systemd` daemon.

Create a file at `/etc/systemd/system/vllm.service`:
```ini
[Unit]
Description=vLLM OpenAI Compatible Server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/vedaz_ai_engineer
ExecStart=/home/ubuntu/vedaz_ai_engineer/venv/bin/python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-3B-Instruct --enable-lora --lora-modules vedaz-astrologer=/home/ubuntu/vedaz_ai_engineer/outputs --max-lora-rank 16 --gpu-memory-utilization 0.90
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```
Enable and start it via:
```bash
sudo systemctl enable --now vllm
```

### Nginx Reverse Proxy
Never expose port `8000` directly to the internet. Install Nginx and safely proxy inbound traffic from port 80 (or 443) to `127.0.0.1:8000`.

### HTTPS
Use Let's Encrypt (Certbot) in conjunction with Nginx to secure your endpoints with SSL/TLS encryption. Open LLM traffic must be encrypted.

### Monitoring
vLLM automatically exposes Prometheus metrics at the `/metrics` endpoint. Connect this to a Prometheus/Grafana stack to monitor GPU VRAM usage, queue lengths, and request latency.

### Logging
Rely on `journalctl -u vllm -f` to monitor production logs if using systemd. Alternatively, redirect standard output/error to rotating log files in your systemd configuration.

---

## 13. Troubleshooting

- **Out of Memory (OOM) Errors:** If the server crashes on startup with a CUDA OOM error, lower the memory allocation fraction (e.g., from `0.9` to `0.7`).
- **CUDA Version Mismatch:** vLLM binaries are compiled for specific CUDA versions. Ensure your system's CUDA matches what vLLM expects (verify via `nvidia-smi` and `nvcc --version`).
- **LoRA Rank Errors:** Ensure the `--max-lora-rank` argument in your startup command matches or exceeds the `r` value (e.g., `16`) used during the `SFTTrainer` phase in `train.py`.
- **Adapter Mismatch:** If the model outputs nonsense, ensure the base model explicitly matches the model used during fine-tuning.
