import sounddevice as sd
import scipy.io.wavfile as wav
import subprocess

def play_audio(file):
    subprocess.run(["afplay", file])