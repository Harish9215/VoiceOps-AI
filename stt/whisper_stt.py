from faster_whisper import WhisperModel

import config

# Whisper tends to emit these on near-silence; drop segments that are very likely noise.
NO_SPEECH_PROB_LIMIT = 0.6


class WhisperSTT:
    def __init__(self, model_size: str = config.WHISPER_MODEL):
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_path) -> str:
        segments, _info = self.model.transcribe(
            str(audio_path),
            language=config.WHISPER_LANGUAGE,
            vad_filter=True,                       # skip non-speech so silence isn't "transcribed"
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,      # avoids repetition/hallucination loops
        )
        parts = [seg.text.strip() for seg in segments if seg.no_speech_prob < NO_SPEECH_PROB_LIMIT]
        return " ".join(p for p in parts if p).strip()
