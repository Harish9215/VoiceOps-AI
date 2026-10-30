"""Cross-platform WAV playback (works on macOS, Linux and Windows)."""
import wave

import numpy as np


def play_audio(file) -> None:
    import sounddevice as sd

    with wave.open(str(file), "rb") as f:
        rate, channels, width = f.getframerate(), f.getnchannels(), f.getsampwidth()
        frames = f.readframes(f.getnframes())
    if width != 2:
        raise ValueError(f"Unsupported WAV sample width: {width * 8}-bit (expected 16-bit)")
    data = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        data = data.reshape(-1, channels)
    sd.play(data, rate)
    sd.wait()
