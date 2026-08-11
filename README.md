# Scriptdump

A Streamlit-based web application for transcribing audio files using OpenAI Whisper models. Upload audio files, select a model, and get text transcriptions with an interactive UI displaying waveform and spectrogram visualizations.

## Technical Description

Scriptdump is a Python application built on top of Streamlit that leverages OpenAI's Whisper models for automatic speech recognition (ASR). The application provides a graphical interface for audio transcription with support for multiple Whisper model sizes and output formats.

### Architecture

```
scriptdump/
├── src/scriptdump/
│   ├── app.py                    # Streamlit app entry point, page navigation
│   ├── __main__.py               # CLI entry point (scriptdump command)
│   ├── pages/
│   │   └── whisper_audio.py      # Whisper transcription UI page
│   ├── libs/
│   │   ├── settings.py           # Configuration loader (YAML-based)
│   │   ├── huggingface.py        # HuggingFace model download utility
│   │   └── utils/
│   │       ├── __init__.py       # Accelerator detection (CUDA/MPS/CPU)
│   │       ├── audio_pipelines.py # Audio resampling, waveform & spectrum rendering
│   │       ├── console_utils.py  # ANSI colored console output
│   │       └── parameters.py     # YAML config wrapper class
│   └── parameters.yaml           # Default configuration
├── pyproject.toml                # Project metadata and dependencies
└── README.md
```

### Key Components

**Audio Processing Pipeline**: Audio files are decoded using `torchcodec` and resampled to 16 kHz mono for Whisper inference. Audio longer than 30 seconds is automatically split into 30-second chunks with 2-second overlap to improve transcription quality, then results are merged with globally-consistent timestamps. Stereo files are automatically converted to mono. The pipeline supports waveform and spectrogram visualization via `matplotlib`, with audio preview in the browser.

**Model Management**: Whisper checkpoints are downloaded from HuggingFace Hub using `huggingface-hub`. Models are cached locally and loaded with `transformers` pipeline (`automatic-speech-recognition`). The app supports multiple model variants: Tiny, Small, Medium, and Large.

**Hardware Acceleration**: The application auto-detects available compute backends: CUDA (GPU with `float16`), MPS (Apple Silicon with `float32`), or CPU fallback. Models are loaded with `low_cpu_mem_usage` option to reduce VRAM consumption.

**Configuration**: App settings are loaded from `parameters.yaml` via the `Properties` class, which wraps YAML data into an attribute-accessible namespace.

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| streamlit | >=1.61.1 | Web UI framework |
| torch | >=2.13.0 | Tensor computation |
| torchaudio | >=2.11.0 | Audio I/O and transforms |
| torchcodec | >=0.15.0 | Audio decoding |
| transformers | >=4.40.0 | Whisper model pipeline |
| huggingface-hub | >=1.27.0 | Model checkpoint downloads |
| pyyaml | >=6.0 | Configuration parsing |
| python-dotenv | >=1.2.2 | Environment variable loading |
| matplotlib | >=3.7.0 | Waveform/spectrogram rendering |

## Installation

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone <repo-url>
cd scriptdump

# Create virtual environment and install dependencies
uv sync
```

## Configuration

Copy and customize the default configuration:

```bash
cp src/scriptdump/parameters.yaml src/scriptdump/parameters.yaml.local
```

Edit `parameters.yaml.local` to set your preferred model, HuggingFace API token, and cache directories.

### Configuration Fields

| Field | Description | Default |
|---|---|---|
| `whisper.model` | Default model to use | `"Whisper Tiny"` |
| `whisper.supported_models` | Available models (display name -> HuggingFace repo ID) | 4 models |
| `whisper.revision` | Model branch/revision | `"main"` |
| `huggingface.apitoken` | HuggingFace access token for gated models | `your_api_token` |
| `huggingface.local_dir` | Local path for downloaded model files | `/tmp/cache/huggingface/local_repo` |
| `huggingface.cache_dir` | HuggingFace cache directory | `/tmp/cache/huggingface` |

## Usage

### Start the app

```bash
# Using the installed scriptdump command
scriptdump

# Or run directly
uv run streamlit run src/scriptdump/app.py
```

The application will start a Streamlit server and open in your default browser (typically at `http://localhost:8501`).

### Using the Web Interface

1. **Select a model** — Choose from Whisper Tiny, Small, Medium, or Large in the sidebar. The model will be downloaded from HuggingFace on first use and cached for subsequent runs.

2. **Upload an audio file** — Use the file uploader to select a `.mp3` or `.wav` file. Audio longer than 30 seconds is automatically split into 30-second chunks with 2-second overlap for better accuracy. Stereo files are automatically converted to mono.

3. **Configure inference parameters** — Set the source language (English, German, French, Italian, Spanish), choose between Transcribe or Translate tasks, and enable timestamped output for clips longer than 30 seconds.

4. **Preview audio** — Expand the "Clip Information" panel to view waveform and spectrogram visualizations, and listen to an audio preview.

5. **Transcribe** — Click "Transcribe Audio" to run the inference. The transcription result is displayed with an option to download as a `.txt` file.

### Options

- **Low VRAM Settings** — Reduces model memory footprint by using `low_cpu_mem_usage`. Enable this if you encounter out-of-memory errors.
- **Return Timestamps** — Automatically enabled for audio longer than 30 seconds. Outputs chunk-level time boundaries in JSON format.

## Environment Variables

Set via a `.env` file in the app directory:

| Variable | Description |
|---|---|
| `CONFIG_FILE` | Override the default YAML config path |
| `HF_TOKEN` | HuggingFace API token (for downloading gated models) |

## Supported Audio Formats

- `mp3`
- `wav`

Files are automatically resampled to 16 kHz mono regardless of input format.
