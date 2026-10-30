import shutil
import subprocess

import config


class TTSError(RuntimeError):
    pass


def speak(text: str):
    """Synthesize `text` to a WAV file with Piper and return its path."""
    if not text.strip():
        raise TTSError("Nothing to speak.")
    piper = shutil.which("piper")
    if piper is None:
        raise TTSError("`piper` not found on PATH. Run: pip install -r requirements.txt")
    if not config.PIPER_MODEL.exists():
        raise TTSError(
            f"Piper model not found at {config.PIPER_MODEL}. "
            "Run: python scripts/download_models.py"
        )

    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    out = config.WORK_DIR / "response.wav"
    try:
        subprocess.run(
            [piper, "--model", str(config.PIPER_MODEL), "--output_file", str(out)],
            input=text,
            text=True,
            capture_output=True,
            check=True,
            timeout=config.TTS_TIMEOUT,
        )
    except subprocess.CalledProcessError as e:
        raise TTSError(f"Piper failed (exit {e.returncode}): {(e.stderr or '').strip()}") from e
    except subprocess.TimeoutExpired as e:
        raise TTSError(f"Piper timed out after {config.TTS_TIMEOUT:.0f}s") from e
    return out
