"""One question per call to a vision model. Port of notebooks/groq.ipynb, pointed at xAI.

Integration seam: depend on the `Perception` protocol, not on `GrokPerception`.
Tests inject a fake; the provider can be swapped without touching callers.
"""

import asyncio
import base64
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import httpx
from loguru import logger

from src.config import (
    XAI_ATTEMPT_TIMEOUT,
    XAI_PRICE_COMPLETION,
    XAI_PRICE_PROMPT,
    XAI_REQUEST_TIMEOUT,
)

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
class Usage:
    """Tokens billed by the API. Mutable accumulator: `total += call`."""

    prompt: int = 0
    completion: int = 0
    calls: int = 0

    def __iadd__(self, other: "Usage") -> "Usage":
        self.prompt += other.prompt
        self.completion += other.completion
        self.calls += other.calls
        return self

    @property
    def usd(self) -> float:
        return (
            self.prompt * XAI_PRICE_PROMPT + self.completion * XAI_PRICE_COMPLETION
        ) / 1_000_000

    def as_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "total": self.prompt + self.completion,
            "calls": self.calls,
            "usd": round(self.usd, 6),
        }

    def __str__(self) -> str:
        return (
            f"{self.prompt}+{self.completion} tokens, "
            f"{self.calls} calls, ${self.usd:.4f}"
        )


@dataclass
class Rule:
    predicate: str
    direction: str  # rising | falling
    is_transition: bool
    usage: Usage = field(default_factory=Usage)


@dataclass
class Observation:
    state: bool
    evidence: str
    usage: Usage = field(default_factory=Usage)


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
        self._keys = keys
        self._active = 0  # sticky: the first key is primary, later ones are fallbacks
        self.client = httpx.AsyncClient(base_url=BASE_URL, timeout=30)

    async def aclose(self) -> None:
        await self.client.aclose()

    async def _ask(self, content: Any) -> tuple[str, Usage]:
        try:
            return await asyncio.wait_for(
                self._ask_with_failover(content), XAI_REQUEST_TIMEOUT
            )
        except asyncio.TimeoutError as e:
            raise PerceptionError("xAI request deadline exceeded") from e

    async def _ask_with_failover(self, content: Any) -> tuple[str, Usage]:
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 96,
            "messages": [{"role": "user", "content": content}],
        }
        last: Optional[Exception] = None
        for _ in self._keys:  # try each key at most once per call
            key = self._keys[self._active]
            try:
                response = await asyncio.wait_for(
                    self.client.post(
                        "/chat/completions",
                        json=payload,
                        headers={"Authorization": f"Bearer {key}"},
                    ),
                    XAI_ATTEMPT_TIMEOUT,
                )
                if response.status_code in (429, 500, 502, 503):
                    last = PerceptionError(
                        f"xai {response.status_code}: {response.text[:120]}"
                    )
                    self._failover(last)
                    continue
                response.raise_for_status()
                body = response.json()
                u = body.get("usage") or {}
                usage = Usage(
                    int(u.get("prompt_tokens", 0)),
                    int(u.get("completion_tokens", 0)),
                    1,
                )
                return body["choices"][0]["message"].get("content") or "", usage
            except (httpx.HTTPError, asyncio.TimeoutError) as e:
                last = e
                self._failover(e)
        raise PerceptionError(f"{type(last).__name__}: {last}" if last else "no keys")

    def _failover(self, error: Exception) -> None:
        if len(self._keys) == 1:
            logger.warning("xai key failed: {!r}", error)
            return
        self._active = (self._active + 1) % len(self._keys)
        logger.warning(
            "xai key failed: {!r}; switching to key #{}", error, self._active + 1
        )

    async def normalize(self, rule: str) -> Rule:
        raw, usage = await self._ask(NORMALIZE_PROMPT.format(rule=rule))
        data = _json(raw)
        direction = data.get("direction")
        if direction not in ("rising", "falling") or not data.get("predicate"):
            raise PerceptionError(f"cannot parse rule from: {raw[:120]}")
        return Rule(
            str(data["predicate"]),
            direction,
            bool(data.get("is_transition", True)),
            usage,
        )

    async def detect(self, jpeg: bytes, predicate: str) -> Observation:
        image = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        raw, usage = await self._ask(
            [
                {"type": "text", "text": DETECT_PROMPT.format(predicate=predicate)},
                {"type": "image_url", "image_url": {"url": image}},
            ]
        )
        data = _json(raw)
        if "state_now" in data:
            return Observation(
                bool(data["state_now"]), str(data.get("evidence", "")), usage
            )
        found = re.findall(r"true|false", raw.lower())
        if not found:
            raise PerceptionError(f"no verdict in: {raw[:120]}")
        return Observation(found[-1] == "true", "<unparsed>", usage)
