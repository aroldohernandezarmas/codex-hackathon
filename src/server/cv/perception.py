"""One question per call to a vision model. Port of notebooks/groq.ipynb, pointed at xAI.

Integration seam: depend on the `Perception` protocol, not on `GrokPerception`.
Tests inject a fake; the provider can be swapped without touching callers.
"""

import base64
import json
import re
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Iterator, Optional, Protocol

import httpx
from loguru import logger

BASE_URL = "https://api.x.ai/v1"

DETECT_PROMPT = (
    "You look at a single still frame from a fixed security camera.\n"
    "Answer only this about THIS frame: is the following true right now?\n"
    "  {predicate}\n"
    "If the frame is too dark or unclear to tell, answer false.\n"
    'Reply with JSON only: {{"state_now": true|false, '
    '"evidence": "<a few words on what you see>"}}'
)

NORMALIZE_PROMPT = (
    "A user wants a camera to notify them when something HAPPENS. Their words:\n"
    "  {rule}\n"
    "Rewrite it as a STATE that is either true or false in a single still frame, "
    "plus the direction of the change the user is waiting for.\n"
    "- predicate: a short present-tense sentence about the scene, "
    "e.g. 'a cat is on the table'\n"
    "- direction: 'rising' if the user waits for the predicate to become true "
    "(appears, arrives, jumps on, turns on), 'falling' if they wait for it to "
    "become false (leaves, goes away, turns off)\n"
    "- is_transition: false if the words describe no change at all "
    "(just an object or a scene)\n"
    'Reply with JSON only: {{"predicate": "...", '
    '"direction": "rising"|"falling", "is_transition": true|false}}'
)


class PerceptionError(Exception):
    pass


@dataclass
class Rule:
    predicate: str
    direction: str  # rising | falling
    is_transition: bool


@dataclass
class Observation:
    state: bool
    evidence: str


class Perception(Protocol):
    async def normalize(self, rule: str) -> Rule: ...

    async def detect(self, jpeg: bytes, predicate: str) -> Observation: ...


def _json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        found = re.search(r"\{.*\}", raw, re.S)
        if found:
            try:
                return json.loads(found.group())
            except json.JSONDecodeError:
                pass
        return {}


class GrokPerception:
    def __init__(self, keys: list[str], model: str) -> None:
        if not keys:
            raise PerceptionError("no XAI_API_KEYS configured")
        self.model = model
        self._keys: Iterator[str] = cycle(keys)
        self._retries = min(len(keys), 2)  # one retry on the next key, if any
        self.client = httpx.AsyncClient(base_url=BASE_URL, timeout=30)

    async def aclose(self) -> None:
        await self.client.aclose()

    async def _ask(self, content: Any) -> str:
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 96,
            "messages": [{"role": "user", "content": content}],
        }
        last: Optional[Exception] = None
        for _ in range(self._retries):
            key = next(self._keys)
            try:
                response = await self.client.post(
                    "/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {key}"},
                )
                if response.status_code in (429, 500, 502, 503):
                    last = PerceptionError(
                        f"xai {response.status_code}: {response.text[:120]}"
                    )
                    logger.warning("{}; retrying on next key", last)
                    continue
                response.raise_for_status()
                return response.json()["choices"][0]["message"].get("content") or ""
            except httpx.HTTPError as e:
                last = e
                logger.warning("xai request failed: {}", e)
        raise PerceptionError(str(last))

    async def normalize(self, rule: str) -> Rule:
        raw = await self._ask(NORMALIZE_PROMPT.format(rule=rule))
        data = _json(raw)
        direction = data.get("direction")
        if direction not in ("rising", "falling") or not data.get("predicate"):
            raise PerceptionError(f"cannot parse rule from: {raw[:120]}")
        return Rule(
            str(data["predicate"]), direction, bool(data.get("is_transition", True))
        )

    async def detect(self, jpeg: bytes, predicate: str) -> Observation:
        image = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        raw = await self._ask(
            [
                {"type": "text", "text": DETECT_PROMPT.format(predicate=predicate)},
                {"type": "image_url", "image_url": {"url": image}},
            ]
        )
        data = _json(raw)
        if "state_now" in data:
            return Observation(bool(data["state_now"]), str(data.get("evidence", "")))
        found = re.findall(r"true|false", raw.lower())
        if not found:
            raise PerceptionError(f"no verdict in: {raw[:120]}")
        return Observation(found[-1] == "true", "<unparsed>")
