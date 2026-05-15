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

The repo ships a [`requirements.txt`](nanovllm-voxcpm-main/requirements.txt) at `nanovllm-voxcpm-main/`. It pins `torch==2.7.1` (cu126), a prebuilt `flash-attn==2.8.3` wheel, and all FastAPI service deps. Plain `pip install` works — no source compile.

```bash
cd nanovllm-voxcpm-main

# (recommended) work in a virtualenv
python3.10 -m venv .venv
source .venv/bin/activate

# Production install
pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps .              # installs the nano-vllm-voxcpm engine package
```

Add `pip install -e ./deployment` (or just keep the source tree on `PYTHONPATH`) if you want to import the FastAPI service modules outside of running `fastapi run`.

Dev tools (linters / tests):

```bash
pip install -r requirements-dev.txt
```

Docker (CUDA 12.6 runtime base — no compile, ~3-5 min build):

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

### Image size & caveats

- Disk: ~10-12 GB for the final image (torch + cuda runtime libs).
- Build disk: ~15 GB free in `/var/lib/docker`. Check with `df -h /var/lib/docker`.
- The `requirements.txt` layer is cached on rebuilds, so only changes to that file (or earlier layers) trigger a full re-install.

### Updating torch / flash-attn

If you bump `torch` in `requirements.txt`, you must also swap the flash-attn URL to a matching wheel. The filename pattern is:

```
flash_attn-<flashver>+cu<series>torch<torchver>cxx11abi<TRUE|FALSE>-cp<py>-cp<py>-linux_x86_64.whl
```

List available wheels:

```bash
curl -s https://api.github.com/repos/Dao-AILab/flash-attention/releases/tags/v2.8.3 \
  | grep browser_download_url
```

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

From the `nanovllm-voxcpm-main/` directory (with the venv activated):

```bash
# Preset library
export VOICE_PRESETS_DIR="$(cd .. && pwd)/voice_presets"

# VoxCPM2 checkpoint
export NANOVLLM_MODEL_PATH=/path/to/VoxCPM2

# Run the service (PYTHONPATH so `app.*` imports work without installing deployment)
PYTHONPATH=deployment fastapi run deployment/app/main.py --host 0.0.0.0 --port 8000
```

Equivalent uvicorn invocation (matches the container entrypoint):

```bash
cd deployment
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000
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
