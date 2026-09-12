"""Minimal check: send one image to xAI Grok vision, print the answer.

Usage: poetry run python scripts/xai_check.py [image_path] [predicate]
"""

import base64
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import XAI_API_KEYS, XAI_MODEL  # noqa: E402

URL = "https://api.x.ai/v1/chat/completions"
PROMPT = (
    "You look at a single still frame from a fixed security camera.\n"
    "Answer only this about THIS frame: is the following true right now?\n"
    "  {predicate}\n"
    "If the frame is too dark or unclear to tell, answer false.\n"
    'Reply with JSON only: {{"state_now": true|false, "evidence": "<a few words on what you see>"}}'
)


def main() -> int:
    assert XAI_API_KEYS, "XAI_API_KEYS is empty"
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/1.png")
    predicate = sys.argv[2] if len(sys.argv) > 2 else "a cat is on the table"

    image_b64 = base64.b64encode(image_path.read_bytes()).decode()
    payload = {
        "model": XAI_MODEL,
        "temperature": 0,
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT.format(predicate=predicate)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
    }
    response = httpx.post(
        URL,
        headers={"Authorization": f"Bearer {XAI_API_KEYS[0]}"},
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    print(f"model={XAI_MODEL} image={image_path.name} predicate={predicate!r}")
    print(content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
