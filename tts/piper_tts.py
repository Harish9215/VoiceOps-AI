import subprocess

MODEL_PATH = "tts_models/en_US-amy-medium.onnx"
OUTPUT_FILE = "response.wav"

def speak(text):
    command = [
        "piper",
        "--model", MODEL_PATH,
        "--output_file", OUTPUT_FILE
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    process.stdin.write(text)
    process.stdin.close()
    process.wait()

    return OUTPUT_FILE