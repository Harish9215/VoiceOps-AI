"""Central configuration. Every value can be overridden with an environment variable."""
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- LLM (Ollama) ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral-nemo")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "60"))  # seconds
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))  # excl. system prompt
MAX_SUMMARY_CHARS = int(os.getenv("MAX_SUMMARY_CHARS", "12000"))

# --- STT ---
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en") or None  # "" -> auto-detect

# --- TTS (Piper) ---
PIPER_MODEL = Path(os.getenv("PIPER_MODEL", ROOT / "tts_models" / "en_US-amy-medium.onnx"))
TTS_TIMEOUT = float(os.getenv("TTS_TIMEOUT", "60"))

# --- Audio / voice activity detection ---
SAMPLE_RATE = 16000
BLOCK_MS = 30                # analysis frame size
SILENCE_END_MS = int(os.getenv("SILENCE_END_MS", "800"))     # silence that ends an utterance
START_TIMEOUT_S = float(os.getenv("START_TIMEOUT_S", "8"))   # give up if nobody speaks
MAX_UTTERANCE_S = float(os.getenv("MAX_UTTERANCE_S", "30"))  # hard cap per turn
MIN_SPEECH_MS = int(os.getenv("MIN_SPEECH_MS", "250"))       # ignore clicks/pops
MIN_RMS_THRESHOLD = float(os.getenv("MIN_RMS_THRESHOLD", "300"))  # int16 scale

# Recordings live in a temp dir (never in the repo) and are deleted after use.
WORK_DIR = Path(tempfile.gettempdir()) / "voiceops-ai"

# --- Conversation ---
# Explicit phrases only. "thanks"/"thank you" are intentionally NOT here.
EXIT_PHRASES = ["goodbye", "bye bye", "bye", "end the call", "end call", "hang up"]
