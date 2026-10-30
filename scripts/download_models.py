"""Download the default Piper voice (en_US-amy-medium) into tts_models/."""
import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium"
FILES = ["en_US-amy-medium.onnx", "en_US-amy-medium.onnx.json"]
DEST = Path(__file__).resolve().parent.parent / "tts_models"


def main() -> int:
    DEST.mkdir(exist_ok=True)
    for name in FILES:
        target = DEST / name
        if target.exists() and target.stat().st_size > 0:
            print(f"✓ {name} already present")
            continue
        print(f"↓ Downloading {name} ...")
        try:
            urllib.request.urlretrieve(f"{BASE}/{name}", target)
        except Exception as e:  # noqa: BLE001
            target.unlink(missing_ok=True)
            print(f"✗ Failed to download {name}: {e}", file=sys.stderr)
            return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
