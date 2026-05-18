# VoxCPM2 Inference Server

FastAPI inference server for [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) — a multilingual voice cloning model — powered by [nano-vLLM](https://github.com/a710128/nanovllm-voxcpm).

Features:
- Streaming MP3 audio generation via `POST /generate`
- Real-time browser playback using the MediaSource API
- Voice preset library (13 languages, multiple voices and emotions)
- Upload your own reference audio for voice cloning
- Zero-shot generation (no reference audio required)
- LoRA adapter management at runtime
- Prometheus metrics, health/readiness probes
- Single Dockerfile that works on any CUDA GPU (Volta → Hopper)

---

## Quick start with Docker (recommended)

### 1. Build the image

Build context is the repo root. The `flash-attn` extension compiles from source, so this takes **20–40 minutes** on first build.

```bash
docker build -t voxcpm2-demo:latest .
```

To restrict to your GPU and speed up the build, pass your compute capability:

| GPU family         | Example cards           | `TORCH_CUDA_ARCH_LIST` |
|--------------------|-------------------------|------------------------|
| Volta              | V100                    | `"7.0"`                |
| Turing             | T4, RTX 2080            | `"7.5"`                |
| Ampere (data center)| A100, A30              | `"8.0"`                |
| Ampere (consumer)  | RTX 3090, A40, A10      | `"8.6"`                |
| Ada Lovelace       | **L4**, RTX 4090        | `"8.9"`                |
| Hopper             | H100, H200              | `"9.0"`                |

```bash
# L4 GPU (this machine)
docker build --build-arg TORCH_CUDA_ARCH_LIST="8.9" -t voxcpm2-demo:latest .

# Multiple targets
docker build --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" -t voxcpm2-demo:latest .
```

### 2. Download the model

The model must be present on the host before starting the container.

```bash
# Via HuggingFace Hub CLI
pip install huggingface_hub
huggingface-cli download openbmb/VoxCPM2 --local-dir /models/VoxCPM2

# Or via Python
python3 - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="openbmb/VoxCPM2", local_dir="/models/VoxCPM2")
EOF
```

### 3. Run the container

```bash
docker run --rm \
  --gpus all \
  -p 8000:8000 \
  -v /models/VoxCPM2:/models/VoxCPM2:ro \
  voxcpm2-demo:latest
```

Then open **http://localhost:8000/ui** in your browser.

#### Common run options

```bash
# Use a specific GPU (e.g. GPU index 1)
docker run --rm --gpus '"device=1"' \
  -p 8000:8000 \
  -v /models/VoxCPM2:/models/VoxCPM2:ro \
  voxcpm2-demo:latest

# Cache HuggingFace downloads between runs
docker run --rm --gpus all \
  -p 8000:8000 \
  -v /models/VoxCPM2:/models/VoxCPM2:ro \
  -v /tmp/hf-cache:/var/cache/nanovllm/hf \
  voxcpm2-demo:latest

# Let the container download the model itself (requires internet + HF token)
docker run --rm --gpus all \
  -p 8000:8000 \
  -e NANOVLLM_MODEL_PATH=openbmb/VoxCPM2 \
  -e HF_TOKEN=hf_your_token_here \
  -v /tmp/hf-cache:/var/cache/nanovllm/hf \
  voxcpm2-demo:latest

# Tune GPU memory and concurrency
docker run --rm --gpus all \
  -p 8000:8000 \
  -v /models/VoxCPM2:/models/VoxCPM2:ro \
  -e NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION=0.90 \
  -e NANOVLLM_SERVERPOOL_MAX_NUM_SEQS=8 \
  voxcpm2-demo:latest
```

---

## Running without Docker (local dev)

### Prerequisites

- Linux + NVIDIA GPU (CUDA 12+)
- Python 3.10–3.12
- `uv` — install with `pip install uv`
- `ninja`, `build-essential`, `ffmpeg`, `libsndfile1-dev`

```bash
sudo apt-get install -y ninja-build build-essential ffmpeg libsndfile1-dev
```

### Install

From the `nanovllm-voxcpm-main/` directory:

```bash
cd nanovllm-voxcpm-main
uv sync --all-packages --no-dev
```

This installs `nano-vllm-voxcpm` and the `deployment` FastAPI service, including `flash-attn` compiled from source.

### Set environment variables

```bash
export NANOVLLM_MODEL_PATH=/models/VoxCPM2      # or openbmb/VoxCPM2 to auto-download
export NANOVLLM_VOICE_PRESETS_DIR=/home/ubuntu/nanovllm-voxcpm2-inference/voice_presets
```

### Start the server

```bash
cd nanovllm-voxcpm-main

uv run fastapi run deployment/app/main.py --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/ui**.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `NANOVLLM_MODEL_PATH` | `/models/VoxCPM2` | Local path or HuggingFace repo id |
| `NANOVLLM_VOICE_PRESETS_DIR` | auto-detected | Path to `voice_presets/` directory |
| `NANOVLLM_SERVERPOOL_DEVICES` | `0` | Comma-separated GPU indices, e.g. `0,1` |
| `NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION` | `0.95` | Fraction of GPU memory to use `(0, 1]` |
| `NANOVLLM_SERVERPOOL_MAX_NUM_SEQS` | `16` | Max concurrent requests |
| `NANOVLLM_SERVERPOOL_MAX_NUM_BATCHED_TOKENS` | `8192` | Max tokens per batch |
| `NANOVLLM_SERVERPOOL_MAX_MODEL_LEN` | `4096` | Max sequence length |
| `NANOVLLM_SERVERPOOL_ENFORCE_EAGER` | `false` | Disable CUDA graphs (useful for debugging) |
| `NANOVLLM_MP3_BITRATE_KBPS` | `192` | MP3 output bitrate |
| `NANOVLLM_MP3_QUALITY` | `2` | LAME quality preset (`0` = best, `2` = fast) |
| `NANOVLLM_LORA_ENABLED` | `false` | Enable runtime LoRA adapter management |
| `HF_TOKEN` | — | HuggingFace token (if model requires authentication) |
| `HF_HOME` | `/var/cache/nanovllm/hf` | HuggingFace cache directory |

---

## API reference

Interactive docs are at **http://localhost:8000/docs**.

### Health probes

```
GET /health   →  {"status": "ok"}                   (liveness)
GET /ready    →  200 once model is loaded            (readiness)
```

### Model info

```
GET /info
```

### Generate speech (streaming MP3)

```
POST /generate
Content-Type: application/json

{
  "target_text": "Hello, world.",
  "cfg_value": 1.5,
  "temperature": 1.0,
  "max_generate_length": 2000,

  // Voice reference — pick ONE of the three forms below:

  // Form 1 — Voice preset (ref audio, no transcript)
  "ref_audio_wav_base64": "<base64-encoded WAV bytes>",
  "ref_audio_wav_format": "wav",

  // Form 2 — Custom audio with transcript (precise cloning)
  "prompt_wav_base64": "<base64-encoded WAV bytes>",
  "prompt_wav_format": "wav",
  "prompt_text": "Exact words spoken in the reference audio.",

  // Form 3 — Zero-shot (omit all audio fields)
}
```

Response: `audio/mpeg` stream with headers `X-Audio-Sample-Rate` and `X-Audio-Channels`.

### Encode audio to latents (optional, for caching)

```
POST /encode_latents
Content-Type: application/json

{
  "wav_base64": "<base64-encoded audio file bytes>",
  "wav_format": "wav"
}
```

Returns `prompt_latents_base64` that can be reused across requests instead of re-encoding the same reference audio.

### Voice presets

```
GET /voice_presets
```

Returns the full preset library:

```json
{
  "presets": {
    "en": {
      "woman_voice_wise": [
        {
          "emotion": "happy",
          "wav": "/voice_presets/audio/en/woman_voice_wise/en_woman_happy.wav",
          "mp3": "/voice_presets/audio/en/woman_voice_wise/en_woman_happy.mp3"
        }
      ]
    }
  }
}
```

Audio files are served directly from `GET /voice_presets/audio/<lang>/<voice_type>/<file>`.

---

## Project layout

```
nanovllm-voxcpm2-inference/
├── Dockerfile                        # Single-image build (any CUDA GPU)
├── .dockerignore
├── README.md                         # This file
├── voice_presets/                    # Built-in reference audio library
│   ├── en/
│   │   ├── woman_voice_wise/
│   │   │   ├── en_woman_happy.wav
│   │   │   ├── en_woman_happy.mp3
│   │   │   └── ...
│   │   └── ...
│   └── <ar|de|es|fr|hu|it|ja|pl|pt|ru|tr|zh>/
└── nanovllm-voxcpm-main/             # Core inference engine + FastAPI service
    ├── nanovllm_voxcpm/              # Python package (scheduler, KV cache, model)
    ├── deployment/
    │   ├── app/
    │   │   ├── main.py               # FastAPI app factory
    │   │   ├── api/routes/
    │   │   │   ├── generate.py       # POST /generate (streaming MP3)
    │   │   │   ├── encode_latents.py # POST /encode_latents
    │   │   │   ├── voice_presets.py  # GET /voice_presets
    │   │   │   ├── health.py
    │   │   │   ├── info.py
    │   │   │   ├── lora.py
    │   │   │   └── metrics.py
    │   │   └── static/
    │   │       └── index.html        # Demo UI (streaming playback)
    │   └── Dockerfile                # Deployment-only Dockerfile (no voice_presets)
    └── pyproject.toml
```

---

## Demo UI

The browser UI at **http://localhost:8000/ui** provides:

- **Voice Preset** tab — select language, voice type, and emotion from the built-in library; preview before generating
- **Upload Audio** tab — drag-and-drop your own WAV/MP3/FLAC; add a transcript for precise voice cloning or leave it blank for style-only reference
- **Zero-shot** tab — generate without any voice reference
- Real-time streaming playback via the [MediaSource API](https://developer.mozilla.org/en-US/docs/Web/API/MediaSource) (falls back to buffered playback on unsupported browsers)
- Generation stats: audio duration, generation time, real-time factor (RTF)
- History panel — every generation is saved for the session

---

## Troubleshooting

**Container exits immediately / OOM**
- Reduce `NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION` (e.g. `0.80`)
- Reduce `NANOVLLM_SERVERPOOL_MAX_NUM_SEQS`

**`flash_attn` build fails**
- Ensure `--build-arg TORCH_CUDA_ARCH_LIST` matches your GPU
- Use the `-devel` CUDA base image (the default); `-runtime` is not sufficient

**Model not found**
- Confirm the model directory is mounted at the correct path and contains `config.json`
- Check `docker run` has `:ro` (read-only) or omit it; the model directory must be readable

**`/ready` returns 503 for a long time**
- Model loading can take 2–5 minutes on first start (weight loading + CUDA graph capture)
- Check `docker logs` for progress

**No audio in browser**
- Chrome and Firefox support `audio/mpeg` in MediaSource; Safari may require a page reload after the first generation
