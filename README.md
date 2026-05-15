# nanovllm-voxcpm2-inference

VoxCPM2-only inference server with a browser demo UI for **Ultimate Cloning**, **Controllable Cloning**, and **Voice Design**. Streams MP3 audio chunk-by-chunk: playback starts as soon as the first byte arrives (via the MediaSource API in the browser).

## Layout

```
nanovllm-voxcpm2-inference/
├── nanovllm-voxcpm-main/          # core engine + FastAPI deployment service
│   └── deployment/
│       └── app/
│           ├── api/routes/        # health, info, lora, generate, voices, ui ...
│           ├── static/demo.html   # interactive demo page (served at "/" and "/ui")
│           └── ...
└── voice_presets/                 # preset voice library (mp3 + wav per voice)
    ├── en/  de/  fr/  tr/  ...    # language buckets
    │   └── <voice_name>/
    │       ├── <voice>_<emotion>.{wav,mp3}
    │       └── expressions/       # optional: laugh / cough / chuckle / ...
```

## Features

- **VoxCPM2 only.** Loaded via `nanovllm_voxcpm.VoxCPM.from_pretrained(...)` in the deployment service.
- **Three modes in the demo UI** (`/ui`):
  - **Ultimate Cloning** — provide a voice prompt audio + its transcript; the model faithfully clones that speaker.
  - **Controllable Cloning** — voice prompt audio (speaker) + a separate reference audio (style/expression).
  - **Voice Design** — zero-shot synthesis; the model designs a voice from target text alone.
- **Preset library** mounted from `voice_presets/`. Both `.wav` and `.mp3` files are selectable in the UI for prompt and reference audio.
- **File upload** is also supported in every mode (uploads override the chosen preset).
- **All hyperparameters** exposed in the UI: `cfg_value`, `temperature`, `max_generate_length`, optional `lora_name`.
- **Streaming output playback.** As soon as the first MP3 byte arrives, the browser starts playing via MediaSource. A "TTFB: NN ms" indicator is shown.
- **MP3 download** of the full generated stream once it completes.

## Supported GPUs

The engine depends on **flash-attn v2** (`flash_attn_varlen_func`, `flash_attn_with_kvcache`, `flash_attn_func`) and there is no SDPA / xformers fallback. flash-attn v2 requires NVIDIA **Ampere or newer** (compute capability ≥ 8.0).

| GPU | Compute capability | Supported |
| --- | --- | --- |
| T4 (Turing, sm_75) | 7.5 | **No** — flash-attn won't compile or run |
| V100 (Volta, sm_70) | 7.0 | **No** |
| A10 / A10G / RTX 30-series | 8.6 | Yes |
| A100 | 8.0 | Yes |
| L4 / L40 / RTX 40-series | 8.9 | Yes |
| H100 / H200 | 9.0 | Yes |

On AWS, the cheapest supported instance is `g5.xlarge` (A10G, 24 GB VRAM).

## Build

This repo uses [`uv`](https://docs.astral.sh/uv/) and the deployment service is a uv workspace member.

```bash
cd nanovllm-voxcpm-main
uv sync --all-packages --frozen
```

Or sync just the deployment service:

```bash
cd nanovllm-voxcpm-main
uv sync --package nano-vllm-voxcpm-deployment --frozen
```

Docker (CUDA 13 base image):

```bash
cd nanovllm-voxcpm-main
docker build -f deployment/Dockerfile -t nanovllm-voxcpm2:latest .
```

### Fast Docker build (no from-source compile)

`pyproject.toml` now pins:

- `torch==2.7.1` + `torchaudio==2.7.1` from PyTorch's **cu126** wheel index
- `flash-attn==2.8.3` from its **GitHub release prebuilt wheel** (matches torch 2.7 + cu12 + cp310 + cxx11abi=FALSE)

The Dockerfile base is `nvidia/cuda:12.6.3-runtime-ubuntu22.04` (matches the cu126 wheels; no `devel` toolchain needed since nothing compiles).

Result: a fresh `docker build` is **~3-5 minutes** — just wheel downloads, zero nvcc. Works on `g6.xlarge` (16 GB) without swap.

```bash
cd nanovllm-voxcpm-main
docker build -f deployment/Dockerfile -t nanovllm-voxcpm2:latest .
```

**Host driver requirement:** `nvidia-smi` ≥ **525** (cu12 wheels). AWS DLAMI and `ubuntu` 22.04 AMIs with the recommended NVIDIA driver are fine.

If you ever need to change torch / flash-attn versions, regenerate the matching wheel URL from `https://github.com/Dao-AILab/flash-attention/releases` — the filename is deterministic from torch version + cuda series + python version + cxx11abi flag.

### Legacy: from-source flash-attn build (only if you can't use the prebuilt wheel)

If you must build from source (e.g. an arch the prebuilt wheel doesn't support), remove the `flash-attn` entry from `[tool.uv.sources]` in `pyproject.toml` and switch the base back to `nvidia/cuda:13.0.1-devel-ubuntu22.04`. Then:

The Dockerfile exposes these build args; defaults are conservative (`MAX_JOBS=2`, `NVCC_THREADS=1`) and target ~16 GB RAM hosts:

| Build arg | Default | Notes |
| --- | --- | --- |
| `CUDA_IMAGE` | `nvidia/cuda:13.0.1-devel-ubuntu22.04` | Must match the cu130 PyTorch wheel that uv resolves. Host driver ≥ 580 required. |
| `TORCH_CUDA_ARCH_LIST` | `8.0;8.6;8.9;9.0` | One arch per target GPU. Trimming this is the single biggest speedup. |
| `MAX_JOBS` | `2` | Parallel compile jobs. |
| `NVCC_THREADS` | `1` | Threads per `nvcc` invocation. |

Pick the arch for your GPU:

| GPU | `TORCH_CUDA_ARCH_LIST` |
| --- | --- |
| A100 | `8.0` |
| A10 / A10G | `8.6` |
| L4 / L40 / RTX 40-series | `8.9` |
| H100 / H200 | `9.0` |

Examples:

```bash
# A10G (g5.xlarge / g5.2xlarge, sm_86), 8 vCPU / 32 GB
docker build -f deployment/Dockerfile \
  --build-arg TORCH_CUDA_ARCH_LIST="8.6" \
  --build-arg MAX_JOBS=4 \
  -t nanovllm-voxcpm2:latest .

# A10G on a tight 16 GB host
docker build -f deployment/Dockerfile \
  --build-arg TORCH_CUDA_ARCH_LIST="8.6" \
  --build-arg MAX_JOBS=2 \
  -t nanovllm-voxcpm2:latest .

# L4 (g6.xlarge, sm_89), 16 GB RAM — USE THIS, plain build will OOM
docker build -f deployment/Dockerfile \
  --build-arg TORCH_CUDA_ARCH_LIST="8.9" \
  --build-arg MAX_JOBS=1 \
  -t nanovllm-voxcpm2:latest .

# L4 on g6.2xlarge (32 GB), faster
docker build -f deployment/Dockerfile \
  --build-arg TORCH_CUDA_ARCH_LIST="8.9" \
  --build-arg MAX_JOBS=2 \
  -t nanovllm-voxcpm2:latest .
```

### Add swap before building on small hosts (≤ 16 GB RAM)

A single nvcc job for flash-attn can spike past 8 GB; on a 16 GB EC2 the kernel will OOM-kill the build (or the whole instance) without swap. Run **once on the host** before `docker build`:

```bash
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
free -h   # confirm Swap line is non-zero
```

Persist across reboots:

```bash
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Making rebuilds fast

The Dockerfile uses BuildKit cache mounts for `~/.cache/uv` and `~/.cache/pip`, so flash-attn's expensive compile is reused across builds as long as its inputs (torch / cuda / arch list) don't change. BuildKit is on by default in modern Docker; if you've disabled it, re-enable:

```bash
export DOCKER_BUILDKIT=1
```

The **first** build on a host still pays the full compile cost — that's unavoidable. After that:

1. **Save the image after the first successful build** so you never rebuild on this host:
   ```bash
   docker save nanovllm-voxcpm2:latest | gzip > nanovllm-voxcpm2.tar.gz
   # Restore elsewhere:
   gunzip -c nanovllm-voxcpm2.tar.gz | docker load
   ```
   Or push to ECR / Docker Hub — recommended for k8s.

2. **Resize for the build, then shrink back.** flash-attn compile time is roughly inversely proportional to `MAX_JOBS`. On a 16 GB host you're stuck with `MAX_JOBS=1` (~45-75 min). Stopping the EC2, changing to `g6.2xlarge` (8 vCPU / 32 GB, ~$1/hr) just for the build, then resizing back, costs ~$0.50 and cuts the build to ~12-15 min:
   ```bash
   docker build -f deployment/Dockerfile \
     --build-arg TORCH_CUDA_ARCH_LIST="8.9" \
     --build-arg MAX_JOBS=4 \
     -t nanovllm-voxcpm2:latest .
   ```

### If a previous build froze / OOM'd the instance

After rebooting, clean up wedged Docker state before retrying:

```bash
sudo systemctl restart docker
docker builder prune -af
df -h /var/lib/docker   # need ≥ 25 GB free
```

Then rerun the appropriate `docker build` command above for your GPU.

Free disk requirement during the build: ~20–25 GB. Verify with `free -h` and `df -h /var/lib/docker` before building.

If the host driver is too old for CUDA 13 (`nvidia-smi` reports < 580), override to a CUDA 12.6 base **and** pin torch to cu126 wheels — or use:

```bash
docker build -f deployment/Dockerfile \
  --build-arg CUDA_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu22.04 \
  -t nanovllm-voxcpm2:latest .
```
(only works if the resolved PyTorch wheel is cu126; otherwise the original mismatch returns).

## Configure

Key environment variables (full list in `nanovllm-voxcpm-main/deployment/README.md`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `NANOVLLM_MODEL_PATH` | `~/VoxCPM1.5` | Path to the VoxCPM2 checkpoint. Set to your VoxCPM2 weights dir (or use `openbmb/VoxCPM2` after downloading). |
| `VOICE_PRESETS_DIR` | repo's `voice_presets/` | Root directory for preset voices served at `/voices`. |
| `NANOVLLM_SERVERPOOL_DEVICES` | `0` | Comma-separated GPU ids. |
| `NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION` | `0.95` | KV-cache memory ratio. |
| `NANOVLLM_MP3_BITRATE_KBPS` | `192` | MP3 stream bitrate. |
| `NANOVLLM_MP3_QUALITY` | `2` | LAME quality (0 best … 2 fast). |

## Run

From the `nanovllm-voxcpm-main/` directory:

```bash
# Make sure the preset library is on disk
export VOICE_PRESETS_DIR="$(cd .. && pwd)/voice_presets"

# Point at your VoxCPM2 checkpoint
export NANOVLLM_MODEL_PATH=/path/to/VoxCPM2

uv run fastapi run deployment/app/main.py --host 0.0.0.0 --port 8000
```

Equivalent uvicorn invocation (matches the container entrypoint):

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Docker:

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e NANOVLLM_MODEL_PATH=/models/VoxCPM2 \
  -e VOICE_PRESETS_DIR=/voice_presets \
  -v /path/to/VoxCPM2:/models/VoxCPM2 \
  -v "$(pwd)/voice_presets":/voice_presets \
  nanovllm-voxcpm2:latest
```

## Open the demo UI

After the server reports ready:

- Demo UI: <http://localhost:8000/ui> (also at `/`)
- OpenAPI: <http://localhost:8000/docs>
- Health / Ready: `/health`, `/ready`
- Voice preset listing: `GET /voices`
- Voice preset file (audio bytes): `GET /voices/file?path=en/man_voice_deep/en_man_happy.wav`

In the UI:

1. Pick a tab — **Ultimate Cloning**, **Controllable Cloning**, or **Voice Design**.
2. Type the target text.
3. For cloning modes, pick a preset (language → voice → file) **or** upload your own `.wav` / `.mp3`. Provide the prompt transcript.
4. For **Controllable Cloning**, pick a second reference audio (preset or upload) — this drives style/expression while the prompt drives speaker identity.
5. Adjust `cfg_value`, `temperature`, `max_generate_length` as needed.
6. Press **Generate**. The audio starts playing as soon as the first MP3 byte streams in; once the stream ends a **Download MP3** link appears.

## API summary

- `GET  /voices` — list available preset languages, voices, files, and expression files.
- `GET  /voices/file?path=<rel>` — stream a preset audio file (path is relative to `VOICE_PRESETS_DIR`).
- `POST /generate` — synthesize and stream MP3 (`audio/mpeg`). Supports:
  - prompt source: `prompt_preset` | `prompt_wav_base64` + `prompt_wav_format` | `prompt_latents_base64` (with `prompt_text`)
  - reference source: `ref_audio_preset` | `ref_audio_wav_base64` + `ref_audio_wav_format` | `ref_audio_latents_base64`
  - hyperparameters: `cfg_value`, `temperature`, `max_generate_length`, optional `lora_name`
- `POST /encode_latents` — encode an audio file to prompt latents (for cached prompts).
- `GET  /info`, `GET /health`, `GET /ready`, `GET /metrics` — observability.

See [nanovllm-voxcpm-main/deployment/README.md](nanovllm-voxcpm-main/deployment/README.md) for the full API reference, LoRA management, and Docker / k8s notes.
