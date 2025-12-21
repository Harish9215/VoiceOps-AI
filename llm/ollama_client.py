import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

class OllamaClient:
    def __init__(self, model="mistral-nemo"):
        self.model = model
        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional AI voice assistant handling phone calls. "
                    "Be concise, polite, and conversational."
                )
            }
        ]

    def chat(self, user_text):
        self.messages.append({"role": "user", "content": user_text})

        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload).json()
        assistant_text = response["message"]["content"]

        self.messages.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    def generate_summary(self):
        summary_prompt = (
            "Summarize this phone call.\n"
            "- Caller intent\n"
            "- Key points discussed\n"
            "- Any actions or requests\n"
            "- Overall sentiment\n"
        )

        messages = self.messages + [
            {"role": "user", "content": summary_prompt}
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload).json()
        return response["message"]["content"]