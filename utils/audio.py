import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

def record_audio(filename="input.wav", duration=5, samplerate=16000):
    print("🎙️ Listening...")
    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype=np.int16
    )
    sd.wait()
    wav.write(filename, samplerate, audio)
    print("🛑 Recording finished")