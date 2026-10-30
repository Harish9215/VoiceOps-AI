import sys
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from utils.audio import Endpointer  # noqa: E402


def should_end_call(text):
    # imported here to avoid pulling in whisper/sounddevice at collection time
    import re
    pat = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in config.EXIT_PHRASES) + r")\b")
    return bool(pat.search(text.lower()))


def test_exit_phrases():
    assert should_end_call("Okay, goodbye!")
    assert should_end_call("alright bye")
    assert not should_end_call("Thanks, one more question")
    assert not should_end_call("Thank you.")
    assert not should_end_call("maybe later")


def _frames(level, n):
    return [np.full(480, level, dtype=np.int16) for _ in range(n)]


def test_endpointer_captures_long_speech_then_stops():
    ep = Endpointer()
    for f in _frames(20, 15) + _frames(3000, 300) + _frames(20, 40):  # 9s of speech
        ep.feed(f)
    assert ep.done and not ep.timed_out
    assert len(ep.audio()) > 9 * 16000 * 0.95


def test_endpointer_silence_times_out():
    ep = Endpointer(start_timeout_s=1)
    for f in _frames(20, 100):
        ep.feed(f)
    assert ep.done and ep.timed_out and ep.audio() is None


def test_endpointer_ignores_short_click():
    ep = Endpointer()
    for f in _frames(20, 15) + _frames(3000, 2) + _frames(20, 40):
        ep.feed(f)
    assert ep.audio() is None


def test_llm_history_is_bounded_and_errors_roll_back():
    from llm.ollama_client import LLMError, OllamaClient
    c = OllamaClient()
    with mock.patch.object(OllamaClient, "_post", return_value="ok"):
        for i in range(50):
            c.chat(f"msg {i}")
    assert len(c.history) <= config.MAX_HISTORY_MESSAGES
    assert c.history[0]["role"] == "user"
    assert len(c.transcript) == 100  # full log preserved for the summary
    before = list(c.history); tbefore = list(c.transcript)
    with mock.patch.object(OllamaClient, "_post", side_effect=LLMError("down")):
        try:
            c.chat("hi")
            assert False
        except LLMError:
            pass
    assert c.history == before and c.transcript == tbefore


def test_llm_unreachable_gives_clear_error():
    import requests
    from llm.ollama_client import LLMError, OllamaClient
    with mock.patch("requests.get", side_effect=requests.ConnectionError("refused")):
        try:
            OllamaClient().check_ready()
            assert False
        except LLMError as e:
            assert "ollama serve" in str(e)


def test_tts_checks_return_code(tmp_path):
    import subprocess
    from tts import piper_tts
    model = tmp_path / "m.onnx"
    model.write_text("x")
    with mock.patch.object(config, "PIPER_MODEL", model), \
         mock.patch("shutil.which", return_value="/usr/bin/piper"), \
         mock.patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "piper", stderr="boom")):
        try:
            piper_tts.speak("hello")
            assert False
        except piper_tts.TTSError as e:
            assert "boom" in str(e)
