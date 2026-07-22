# Transcribix — Pinokio

1-click install and launch for [Transcribix](https://github.com/C0m3b4ck/Transcribix) — offline speech-to-text transcription with 11 local AI models.

[![Pinokio](https://img.shields.io/badge/Pinokio-Install-000000?style=for-the-badge&logo=pinokio&logoColor=white)](https://pinokio.co)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

## What This Does

This is a [Pinokio](https://pinokio.co) app that installs and runs [Transcribix](https://github.com/C0m3b4ck/Transcribix) with a single click. It handles:

- **GPU detection** — auto-installs CUDA or CPU PyTorch
- **Model group selection** — install only what you need
- **Isolated venvs** — each model group has its own environment
- **Gradio Web UI** — browser-based interface for transcription

## Installation

### Option 1: Via Pinokio (Recommended)

1. Open Pinokio
2. Click "Discover" → search for "Transcribix"
3. Click "Install" and select a model group
4. Click "Start" to launch the Gradio Web UI

### Option 2: Manual

```bash
git clone https://github.com/C0m3b4ck/Transcribix-Pinokio.git
cd Transcribix-Pinokio

# Install dependencies (pick one group)
pip install -r requirements/whisper.txt      # Whisper variants
pip install -r requirements/nvidia.txt       # NVIDIA models
pip install -r requirements/lightweight.txt  # Lightweight models
pip install -r requirements/all.txt          # All models
pip install -r requirements/gradio.txt       # Web UI

# Install PyTorch (auto-detects GPU)
python torch.js  # or manually: pip install torch torchaudio

# Launch
python app.py
```

Then open `http://localhost:7860`.

## Model Groups

| Group | Models | VRAM | Disk | Best For |
|-------|--------|------|------|----------|
| **Whisper** | faster-whisper, WhisperX, stable-ts, Distil-Whisper, Whisper, whisper.cpp | 2-10 GB | ~8 GB | General use, subtitles |
| **NVIDIA** | Parakeet TDT, Canary Qwen | 2-6 GB | ~5 GB | Best accuracy |
| **Lightweight** | Moonshine, SenseVoice, Vosk | <1 GB | ~2 GB | Low资源 / edge devices |
| **All** | All 11 models | 2-10 GB | ~20 GB | Full comparison |

## Features

- 11 local speech-to-text models — no API keys, no internet
- Word-level timestamps for caption generation
- SRT, VTT, and ASS subtitles with full styling (font, color, size, position, outline, shadow)
- Optional subtitle burning into video via ffmpeg
- Interactive file selection (upload or enter path)
- GPU auto-detection (CUDA/CPU)
- Colorful CLI with system info and position preview

## Gradio Web UI

The web UI provides:

- **File upload** or **file path input** — use either method
- **Model selection** with live info display
- **Subtitle styling** — font, color, size, position, outline, shadow
- **Words per caption** — 1-8 words per subtitle block
- **Optional video burning** — checkbox to burn subtitles into video (requires ffmpeg)
- **Downloadable outputs** — SRT, ASS, and burned video files

```
┌─────────────────────────────────────────────────────────────┐
│  Transcribix                                                │
│  Offline speech-to-text with 11 local AI models             │
│                                                             │
│  Device: CUDA: NVIDIA RTX 3060 (12.0 GB VRAM)              │
│                                                             │
│  ┌─ Input ──────────────┐  ┌─ Subtitle Styling ──────────┐ │
│  │ [Upload File] [Path] │  │ Font: Arial    Size: 24     │ │
│  │ Model: Faster Whisper│  │ Color: White   Pos: Bottom  │ │
│  │ Language: en         │  │ Outline: 2     Shadow: 1    │ │
│  │ [x] Burn into video  │  │ Words/chunk: 3              │ │
│  └──────────────────────┘  └─────────────────────────────┘ │
│                                                             │
│  [  Transcribe & Generate Subtitles  ]                      │
│                                                             │
│  **Transcription complete!**                                │
│  - Model: Faster Whisper                                    │
│  - Words detected: 1234                                     │
│  - Time: 45.2s                                             │
│                                                             │
│  [SRT Subtitles]  [ASS Subtitles]  [Video with Subtitles]  │
└─────────────────────────────────────────────────────────────┘
```

## Supported Formats

| Audio | Video |
|-------|-------|
| WAV, MP3, M4A, FLAC, OGG | MP4, MKV, AVI, MOV, WebM |

## Project Structure

```
Transcribix-Pinokio/
├── pinokio.json               # Pinokio app metadata
├── pinokio.js                 # Pinokio menu system
├── app.py                     # Gradio web UI
├── captioning_comparison.py   # Main engine (11 models)
├── install.js                 # Group-aware installer
├── start.js                   # Launcher
├── torch.js                   # GPU-aware PyTorch installer
├── reset.js                   # Reset venv
├── resetgroup.js              # Reset specific group
├── update.js                  # Update dependencies
├── requirements/
│   ├── whisper.txt            # Whisper variants
│   ├── nvidia.txt             # NVIDIA models
│   ├── lightweight.txt        # Lightweight models
│   ├── all.txt                # All models
│   └── gradio.txt             # Gradio UI
└── venvs/                     # Isolated venvs (created on install)
    ├── whisper/
    ├── nvidia/
    ├── lightweight/
    └── all/
```

## How It Works

1. **Install** — Pinokio creates an isolated venv for your chosen model group
2. **GPU Detection** — `torch.js` checks for CUDA and installs the right PyTorch
3. **Launch** — `start.js` runs `app.py` and opens the Gradio Web UI
4. **Transcribe** — Select a file, pick a model, customize subtitles, click Transcribe
5. **Output** — Download SRT/ASS files or a video with burned-in subtitles

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10+ | 3.11 |
| RAM | 8 GB | 16 GB+ |
| GPU VRAM | 1 GB (Moonshine) | 6 GB+ (large-v3) |
| Disk | 2 GB | 10 GB+ |
| OS | Linux, macOS, Windows | Linux |

## Troubleshooting

**"No module named gradio"**
→ Install gradio: `pip install -r requirements/gradio.txt`

**"ffmpeg not found"**
→ Install ffmpeg: `sudo apt install ffmpeg` (Linux) or `brew install ffmpeg` (macOS)

**"CUDA out of memory"**
→ Use a smaller model (tiny/base) or switch to CPU mode

**Pinokio won't start**
→ Make sure you have Python 3.10+ installed and on your PATH

## Credits

This project wraps [Transcribix](https://github.com/C0m3b4ck/Transcribix) by C0m3b4ck.

### Speech-to-Text Engines

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 Whisper
- [WhisperX](https://github.com/m-bain/whisperX) — Forced phoneme alignment
- [stable-ts](https://github.com/jianfch/stable-ts) — Stabilized timestamps
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) — Parakeet & Canary
- [Moonshine](https://github.com/usefulsensors/moonshine) — Ultra-lightweight ASR
- [FunASR](https://github.com/modelscope/FunASR) — SenseVoice
- [Vosk](https://github.com/alphacep/vosk-api) — Kaldi-based ASR
- [OpenAI Whisper](https://github.com/openai/whisper) — Original implementation
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — C++ port

### Platforms

- [Pinokio](https://pinokio.co) — App launcher platform
- [Gradio](https://github.com/gradio-app/gradio) — Web UI framework

## License

Apache License 2.0
