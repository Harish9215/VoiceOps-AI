from faster_whisper import WhisperModel

class WhisperSTT:
    def __init__(self, model_size="small"):
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"
        )

    def transcribe(self, audio_path):
        segments, _ = self.model.transcribe(audio_path)
        text = " ".join([seg.text for seg in segments])
        return text.strip()