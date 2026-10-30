import re

import config
from llm.ollama_client import LLMError, OllamaClient
from stt.whisper_stt import WhisperSTT
from tts.piper_tts import TTSError, speak
from utils.audio import record_audio
from utils.play_audio import play_audio

# Whole-word match, so "bye" doesn't fire inside other words.
_EXIT_RE = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in config.EXIT_PHRASES) + r")\b")


def should_end_call(text: str) -> bool:
    return bool(_EXIT_RE.search(text.lower()))


def main():
    llm = OllamaClient()
    try:
        llm.check_ready()
    except LLMError as e:
        raise SystemExit(f"❌ {e}")

    stt = WhisperSTT()
    print("📞 Call started. Say 'goodbye' to end (Ctrl+C also works).\n")

    try:
        while True:
            audio_path = record_audio()
            if audio_path is None:
                print("⚠️ No speech detected.")
                continue

            try:
                user_text = stt.transcribe(audio_path)
            finally:
                audio_path.unlink(missing_ok=True)  # don't leave voice recordings lying around

            if not user_text:
                print("⚠️ Couldn't make out any words.")
                continue

            print(f"🧑 Caller: {user_text}")

            if should_end_call(user_text):
                print("📴 Call ended by user.")
                break

            try:
                response = llm.chat(user_text)
            except LLMError as e:
                print(f"⚠️ LLM error: {e}")
                continue
            print(f"🤖 AI: {response}")

            try:
                audio_file = speak(response)
                play_audio(audio_file)
                audio_file.unlink(missing_ok=True)
            except (TTSError, OSError, ValueError) as e:
                print(f"⚠️ Couldn't play the reply aloud ({e}). Continuing with text only.")
    except KeyboardInterrupt:
        print("\n📴 Call interrupted.")

    print("📝 Generating call summary...")
    try:
        print("\n📄 Call Summary:\n", llm.generate_summary())
    except LLMError as e:
        print(f"⚠️ Couldn't generate a summary: {e}")


if __name__ == "__main__":
    main()
