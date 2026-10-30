import requests

import config


class LLMError(RuntimeError):
    pass


SYSTEM_PROMPT = (
    "You are a professional AI voice assistant handling phone calls. "
    "Be concise, polite, and conversational."
)


class OllamaClient:
    def __init__(self, model: str = config.OLLAMA_MODEL, base_url: str = config.OLLAMA_URL):
        self.model = model
        self.chat_url = f"{base_url.rstrip('/')}/api/chat"
        self.tags_url = f"{base_url.rstrip('/')}/api/tags"
        self.system = {"role": "system", "content": SYSTEM_PROMPT}
        self.history: list = []      # rolling window sent to the model (bounded)
        self.transcript: list = []   # full call log, used only for the summary

    # ---- startup check -------------------------------------------------
    def check_ready(self) -> None:
        """Fail fast with a helpful message if Ollama or the model is unavailable."""
        try:
            r = requests.get(self.tags_url, timeout=5)
            r.raise_for_status()
        except requests.RequestException as e:
            raise LLMError(
                f"Can't reach Ollama at {self.tags_url}. Is it running? (`ollama serve`)\n  {e}"
            ) from e
        names = [m.get("name", "") for m in r.json().get("models", [])]
        if not any(n == self.model or n.split(":")[0] == self.model.split(":")[0] for n in names):
            raise LLMError(f"Model '{self.model}' not installed. Run: ollama pull {self.model}")

    # ---- internals -----------------------------------------------------
    def _post(self, messages: list) -> str:
        payload = {"model": self.model, "messages": messages, "stream": False}
        try:
            r = requests.post(self.chat_url, json=payload, timeout=config.OLLAMA_TIMEOUT)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except requests.Timeout as e:
            raise LLMError(f"Ollama timed out after {config.OLLAMA_TIMEOUT:.0f}s") from e
        except requests.RequestException as e:
            raise LLMError(f"Ollama request failed: {e}") from e
        except (KeyError, ValueError) as e:
            raise LLMError(f"Unexpected response from Ollama: {e}") from e

    @staticmethod
    def _trimmed(messages: list) -> list:
        # Keep the most recent messages; make sure the window starts on a user turn.
        window = messages[-config.MAX_HISTORY_MESSAGES:]
        while window and window[0]["role"] != "user":
            window = window[1:]
        return window

    # ---- public API ----------------------------------------------------
    def chat(self, user_text: str) -> str:
        # Build the candidate history first; commit it only if the request succeeds,
        # so a failed call leaves the conversation state untouched.
        candidate = self._trimmed(self.history + [{"role": "user", "content": user_text}])
        reply = self._post([self.system] + candidate)
        self.history = candidate + [{"role": "assistant", "content": reply}]
        self.transcript.append(("Caller", user_text))
        self.transcript.append(("Assistant", reply))
        return reply

    def generate_summary(self) -> str:
        if not self.transcript:
            return "No conversation to summarize."
        text = "\n".join(f"{who}: {msg}" for who, msg in self.transcript)
        text = text[-config.MAX_SUMMARY_CHARS:]  # keep the most recent part if very long
        prompt = (
            "Summarize this phone call.\n"
            "- Caller intent\n- Key points discussed\n- Any actions or requests\n- Overall sentiment\n\n"
            f"Transcript:\n{text}"
        )
        return self._post([self.system, {"role": "user", "content": prompt}])
