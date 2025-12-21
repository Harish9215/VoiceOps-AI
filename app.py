from stt.whisper_stt import WhisperSTT
from llm.ollama_client import OllamaClient
from utils.audio import record_audio
from tts.piper_tts import speak
from utils.play_audio import play_audio

EXIT_PHRASES = ["bye", "goodbye", "thank you", "thanks"]

def should_end_call(text):
    text = text.lower()
    return any(phrase in text for phrase in EXIT_PHRASES)

def main():
    stt = WhisperSTT()
    llm = OllamaClient()

    print("📞 Call started. Say 'bye' to end.\n")

    while True:
        record_audio(duration=5)
        user_text = stt.transcribe("input.wav")

        if not user_text.strip():
            print("⚠️ No speech detected.")
            continue

        print(f"🧑 Caller: {user_text}")

        if should_end_call(user_text):
            print("📴 Call ended by user.")
            break

        response = llm.chat(user_text)
        print(f"🤖 AI: {response}")

        audio_file = speak(response)
        play_audio(audio_file)

    print("📝 Generating call summary...")
    summary = llm.generate_summary()
    print("\n📄 Call Summary:\n", summary)

if __name__ == "__main__":
    main()