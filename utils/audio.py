"""Microphone capture with voice-activity endpointing (stops when you stop talking)."""
import collections
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

import config


def _rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) if frame.size else 0.0


class Endpointer:
    """Energy-based VAD state machine. Feed it fixed-size int16 frames.

    States: waiting for speech -> in speech -> finished.
    Pure logic (no I/O) so it can be unit-tested.
    """

    CALIBRATION_FRAMES = 10  # ~300 ms of ambient noise to set the threshold

    def __init__(
        self,
        block_ms: int = config.BLOCK_MS,
        silence_end_ms: int = config.SILENCE_END_MS,
        start_timeout_s: float = config.START_TIMEOUT_S,
        max_utterance_s: float = config.MAX_UTTERANCE_S,
        min_speech_ms: int = config.MIN_SPEECH_MS,
        min_rms: float = config.MIN_RMS_THRESHOLD,
        preroll_ms: int = 300,
    ):
        self.block_ms = block_ms
        self.silence_frames = max(1, silence_end_ms // block_ms)
        self.start_timeout_frames = int(start_timeout_s * 1000 / block_ms)
        self.max_frames = int(max_utterance_s * 1000 / block_ms)
        self.min_speech_frames = max(1, min_speech_ms // block_ms)
        self.min_rms = min_rms

        self._preroll = collections.deque(maxlen=max(1, preroll_ms // block_ms))
        self._noise: list = []
        self.threshold = min_rms
        self._frames: list = []
        self._speech_frames = 0
        self._silence_run = 0
        self._seen = 0
        self.started = False
        self.done = False
        self.timed_out = False

    def feed(self, frame: np.ndarray) -> None:
        if self.done:
            return
        self._seen += 1
        level = _rms(frame)

        # Calibrate against ambient noise before listening for speech.
        if len(self._noise) < self.CALIBRATION_FRAMES:
            self._noise.append(level)
            if len(self._noise) == self.CALIBRATION_FRAMES:
                self.threshold = max(self.min_rms, float(np.median(self._noise)) * 3.0)
            self._preroll.append(frame)
            return

        is_speech = level >= self.threshold

        if not self.started:
            self._preroll.append(frame)
            if is_speech:
                self.started = True
                self._frames.extend(self._preroll)
                self._speech_frames = 1
            elif self._seen >= self.start_timeout_frames:
                self.done = self.timed_out = True
            return

        self._frames.append(frame)
        if is_speech:
            self._speech_frames += 1
            self._silence_run = 0
        else:
            self._silence_run += 1

        if self._silence_run >= self.silence_frames or len(self._frames) >= self.max_frames:
            self.done = True

    def audio(self) -> Optional[np.ndarray]:
        """Recorded utterance, or None if there was no real speech."""
        if not self.started or self._speech_frames < self.min_speech_frames:
            return None
        return np.concatenate(self._frames)


def record_utterance(samplerate: int = config.SAMPLE_RATE) -> Optional[np.ndarray]:
    """Block until the user finishes one utterance. Returns None on silence."""
    import sounddevice as sd  # imported lazily so the module is importable without audio hardware

    block = int(samplerate * config.BLOCK_MS / 1000)
    ep = Endpointer()
    print("🎙️ Listening...")
    with sd.InputStream(samplerate=samplerate, channels=1, dtype="int16", blocksize=block) as stream:
        while not ep.done:
            data, _overflowed = stream.read(block)
            ep.feed(data[:, 0])
    return ep.audio()


def save_wav(audio: np.ndarray, path: Path, samplerate: int = config.SAMPLE_RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(samplerate)
        f.writeframes(audio.astype(np.int16).tobytes())
    return path


def record_audio(filename: Optional[Path] = None) -> Optional[Path]:
    """Record one utterance to a WAV in the temp work dir. Returns the path, or None if silent."""
    audio = record_utterance()
    if audio is None:
        return None
    path = Path(filename) if filename else config.WORK_DIR / "input.wav"
    print("🛑 Recording finished")
    return save_wav(audio, path)
