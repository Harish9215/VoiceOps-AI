# VoiceOps-AI 🎙️🤖

A small, modular, fully local voice assistant: **speech-to-text → LLM → text-to-speech**, with a call summary at the end.

| Stage | Implementation |
|-------|----------------|
| Listen | `sounddevice` mic capture with voice-activity endpointing (stops when you stop talking) |
| STT | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with its built-in VAD filter |
| LLM | [Ollama](https://ollama.com) (default model: `mistral-nemo`) |
| TTS | [Piper](https://github.com/rhasspy/piper) (default voice: `en_US-amy-medium`) |

Each stage lives in its own module, so you can swap one for another (e.g. a different LLM backend in `llm/`) without touching the rest. Only Ollama is implemented today.

> **Latency note:** the stages run one after another (listen → transcribe → generate → synthesize → play), so expect a pause of a few seconds per turn depending on your hardware. It is not streaming/real-time.

## Project layout

```text
VoiceOps-AI/
├── app.py              # Orchestration loop
├── config.py           # All settings (env-overridable)
├── llm/                # Ollama client (timeouts, bounded history, summary)
├── stt/                # faster-whisper wrapper
├── tts/                # Piper wrapper
├── utils/              # Mic capture + VAD, cross-platform playback
├── scripts/            # download_models.py
├── tests/              # pytest suite
└── tts_models/         # Piper voice files (downloaded, not committed)
```

## Getting started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- A working microphone and speakers
- On Linux you may also need PortAudio (`sudo apt install libportaudio2`) for `sounddevice`

### Install
```bash
git clone https://github.com/Harish9215/VoiceOps-AI.git
cd VoiceOps-AI

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

ollama pull mistral-nemo        # or set OLLAMA_MODEL to a model you already have
python scripts/download_models.py   # downloads the Piper voice into tts_models/
```

### Run
```bash
python app.py
```
Speak naturally; the assistant waits until you pause. Say **"goodbye"**, **"bye"**, **"end the call"** or **"hang up"** (or press Ctrl+C) to finish and get a summary. Phrases like "thanks" or "thank you" do *not* end the call.

## Configuration

Set environment variables to override defaults (see `config.py`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `mistral-nemo` | Chat model |
| `OLLAMA_TIMEOUT` | `60` | Seconds before an LLM call is abandoned |
| `MAX_HISTORY_MESSAGES` | `20` | Messages of context sent to the model |
| `WHISPER_MODEL` | `small` | faster-whisper model size |
| `WHISPER_LANGUAGE` | `en` | Language code, or empty for auto-detect |
| `PIPER_MODEL` | `tts_models/en_US-amy-medium.onnx` | Piper voice |
| `SILENCE_END_MS` | `800` | Silence that ends your turn |
| `MAX_UTTERANCE_S` | `30` | Hard cap per turn |
| `MIN_RMS_THRESHOLD` | `300` | Minimum mic level counted as speech (raise in noisy rooms) |

## Privacy
Recordings are written to your system temp directory and deleted right after use; `*.wav` is git-ignored. Everything runs locally.

## Tests
```bash
pip install pytest
pytest
```

## Contributing
Create a feature branch, add any new dependency to `requirements.txt`, and run `pytest` before opening a PR.

## License
MIT — see [LICENSE](LICENSE).
