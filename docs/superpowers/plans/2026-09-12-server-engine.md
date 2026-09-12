# Server Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A FastAPI server that accepts camera frames per session, gates them locally, asks xAI Grok whether the user's predicate is true in frames that changed, fires an event once on the rising/falling edge, and hands it to a (stubbed) notifier — deployed on Render.

**Architecture:** Sense → perceive → remember → act loop per session. Multi-user by construction: one asyncio process, CPU work (decode, gate) in worker threads under a per-session lock, model calls as background tasks so `POST /frame` never waits on the network. Gate and perception are straight ports of `notebooks/camera_v3.ipynb` and `notebooks/groq.ipynb`. All temporal logic (persist, edge, direction) is deterministic Python in `Tracker`. Sessions live in one in-memory dict with a cap and a TTL. Perception is the single seam (`Perception` protocol) so tests inject a fake and the provider can be swapped.

**Tech Stack:** Python 3.10, Poetry, FastAPI, uvicorn, httpx, numpy, opencv-python-headless, loguru, pytest.

**Spec:** `docs/superpowers/specs/2026-09-12-camera-events-design.md`

## Global Constraints

- Python `^3.10`, dependencies only via `poetry add`; commit `poetry.lock`.
- After every task: `make format`, then `make lint` and `make test` — all green before commit.
- Settings from env in `src/config.py` with defaults; every new variable added to `.env.example`.
- Logging via `loguru`; no `print` in `src/`. (`scripts/` may print — they are operator tools.)
- Never touch `.ipynb` files with file tools; notebooks are read-only reference here.
- Commit messages: no AI attribution lines.
- Gate constants are copied verbatim from `camera_v3.ipynb`: `GATE_SIZE=(128,72)`, `GRID=8`, `THRESHOLD=6.6`, `GLOBAL=0.5`, `PERSIST=2`, `OVERLAP=0.5`, blur σ=2.
- HTTP contract is fixed by the spec; the client plan builds against it in parallel. Do not rename fields.

## File structure

All eight tasks are done on branch `server-engine` (PR #1). Code lives under `src/server/`; the snippets below still say `src/...` — read them as `src/server/...`.

| File | Responsibility |
|---|---|
| `src/config.py` | env → constants (modify) |
| `src/server/cv/gate.py` | frame → `skip/change/light`, anchor + persist logic |
| `src/server/tracker.py` | model answers → confirmed state → fired edge |
| `src/server/cv/perception.py` | `Rule`, `Perception` protocol, `GrokPerception` (httpx, key rotation) |
| `src/server/session.py` | `Session`, `SessionStore` (cap, TTL) |
| `src/server/notifier.py` | `Notifier.notify()` — logs only |
| `src/server/engine.py` | `handle_frame()` — one frame through the loop |
| `src/server/app.py` | FastAPI routes, static files |
| `main.py` | uvicorn entry (modify) |
| `scripts/grok_check.py` | 3-frame / 5-case provider check |
| `tests/test_gate.py`, `tests/test_tracker.py`, `tests/test_session.py`, `tests/test_app.py` | |

---

### Task 1: Dependencies and config

**Files:**
- Modify: `pyproject.toml` (via poetry only), `src/config.py`, `.env.example`
- Test: `tests/test_config.py` (exists — extend)

**Interfaces:**
- Produces: `config.XAI_API_KEYS: list[str]`, `config.XAI_MODEL: str`, `config.MAX_SESSIONS: int`, `config.SESSION_TTL: float`, `config.PORT: int`.

- [ ] **Step 1: Add and move dependencies**

```bash
poetry add fastapi uvicorn httpx python-multipart numpy opencv-python-headless
poetry remove --group dev opencv-python
```

`opencv-python` needs libGL, which Render's box lacks; `-headless` has the same `cv2` API and works for the notebooks too (they never open windows). `pillow` stays in dev — notebooks use it, the server does not.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_config.py`:

```python
import importlib


def test_xai_keys_split_and_strip(monkeypatch):
    monkeypatch.setenv("XAI_API_KEYS", " k1, k2 ,,k3 ")
    from src import config

    importlib.reload(config)
    assert config.XAI_API_KEYS == ["k1", "k2", "k3"]


def test_defaults(monkeypatch):
    for name in ("XAI_API_KEYS", "XAI_MODEL", "MAX_SESSIONS", "SESSION_TTL", "PORT"):
        monkeypatch.delenv(name, raising=False)
    from src import config

    importlib.reload(config)
    assert config.XAI_API_KEYS == []
    assert config.XAI_MODEL == "grok-4.6"
    assert config.MAX_SESSIONS == 10
    assert config.SESSION_TTL == 30.0
    assert config.PORT == 8000
```

- [ ] **Step 3: Run test to verify it fails**

Run: `poetry run pytest tests/test_config.py -v`
Expected: FAIL with `AttributeError: module 'src.config' has no attribute 'XAI_API_KEYS'`

- [ ] **Step 4: Implement**

Append to `src/config.py`:

```python
XAI_API_KEYS = [k.strip() for k in os.getenv("XAI_API_KEYS", "").split(",") if k.strip()]
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4.6")
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "10"))
SESSION_TTL = float(os.getenv("SESSION_TTL", "30"))
PORT = int(os.getenv("PORT", "8000"))
```

Append to `.env.example`:

```
XAI_API_KEYS=
XAI_MODEL=grok-4.6
MAX_SESSIONS=10
SESSION_TTL=30
PORT=8000
```

- [ ] **Step 5: Run tests, format, lint**

Run: `make format && make lint && make test`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml poetry.lock src/config.py .env.example tests/test_config.py
git commit -m "feat: server deps and xAI/session config"
```

---

### Task 2: Grok provider check (needs a real key)

**Files:**
- Create: `scripts/grok_check.py`

**Interfaces:**
- Consumes: `config.XAI_API_KEYS`, `config.XAI_MODEL`, `data/1.png`, `data/2.png`, `data/3.png`.
- Produces: a printed table and a non-zero exit if fewer than 5/5 cases pass. Its request payload and prompt are the reference for Task 5.

This is `notebooks/groq.ipynb` cells 1–4 pointed at xAI. It measures what the notebooks measured on Groq: per-frame accuracy, latency, tokens. Run it for each candidate model; put the fastest one that scores 3/3 and 5/5 into `.env` as `XAI_MODEL`.

- [ ] **Step 1: Write the script**

```python
"""3-frame / 5-case check of a Grok vision model on data/*.png. Usage: poetry run python scripts/grok_check.py [model]"""

import base64
import json
import re
import sys
from io import BytesIO
from pathlib import Path
from time import perf_counter

import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import XAI_API_KEYS, XAI_MODEL  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
MODEL = sys.argv[1] if len(sys.argv) > 1 else XAI_MODEL
URL = "https://api.x.ai/v1/chat/completions"
PREDICATE = "a cat is on the table (a cat on a chair or on the floor does not count)"
TRUTH = {"1.png": False, "2.png": False, "3.png": True}
CASES = [("1.png", "2.png", False), ("2.png", "3.png", True), ("1.png", "3.png", True),
         ("3.png", "3.png", False), ("3.png", "2.png", False)]

PROMPT = (
    "You look at a single still frame from a fixed security camera.\n"
    "Answer only this about THIS frame: is the following true right now?\n"
    "  {predicate}\n"
    "If the frame is too dark or unclear to tell, answer false.\n"
    'Reply with JSON only: {{"state_now": true|false, "evidence": "<a few words on what you see>"}}'
)


def jpeg(name: str, width: int = 640) -> bytes:
    image = Image.open(DATA / name).convert("L")
    image = image.resize((width, round(image.height * width / image.width)))
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=85)
    return buffer.getvalue()


def ask(name: str) -> dict:
    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT.format(predicate=PREDICATE)},
            {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(jpeg(name)).decode()}},
        ]}],
    }
    started = perf_counter()
    response = httpx.post(URL, headers={"Authorization": f"Bearer {XAI_API_KEYS[0]}"}, json=payload, timeout=60)
    response.raise_for_status()
    body = response.json()
    raw = body["choices"][0]["message"].get("content") or ""
    try:
        parsed = json.loads(raw)
        state, evidence = bool(parsed["state_now"]), str(parsed.get("evidence", ""))
    except (json.JSONDecodeError, KeyError, TypeError):
        found = re.findall(r"true|false", raw.lower())
        state, evidence = (found[-1] == "true") if found else None, f"<unparsed: {raw[:60]}>"
    return {"state": state, "evidence": evidence, "seconds": round(perf_counter() - started, 2),
            "tokens": body.get("usage", {}).get("prompt_tokens")}


def main() -> int:
    assert XAI_API_KEYS, "XAI_API_KEYS is empty"
    print(f"model {MODEL}\n")
    state = {}
    for name in TRUTH:
        r = ask(name)
        state[name] = r["state"]
        ok = "ok " if r["state"] == TRUTH[name] else "BAD"
        print(f"{ok} {name}: state={r['state']} ({r['seconds']}s, {r['tokens']} tok) — {r['evidence']}")
    frames = sum(state[n] == TRUTH[n] for n in TRUTH)
    cases = sum(((not state[a]) and state[b]) == expected for a, b, expected in CASES)
    print(f"\nframes {frames}/3 · cases {cases}/5")
    return 0 if frames == 3 and cases == 5 else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run against candidates**

Run, with a real key in `.env`:

```bash
poetry run python scripts/grok_check.py grok-4.6
poetry run python scripts/grok_check.py grok-4.3
poetry run python scripts/grok_check.py grok-4.20-0309-non-reasoning
```

Expected: at least one model prints `frames 3/3 · cases 5/5`. Record seconds per call. Put the fastest passing model into `.env` (`XAI_MODEL=...`) and, if it differs from `grok-4.6`, change the default in `src/config.py` and `.env.example`.

If a model returns HTTP 404, the id is wrong for this key — try the others; the list is at `GET https://api.x.ai/v1/language-models` with the key.

- [ ] **Step 3: Commit**

```bash
git add scripts/grok_check.py src/config.py .env.example
git commit -m "feat: grok provider check on the sample frames"
```

---

### Task 3: Gate

**Files:**
- Create: `src/gate.py`
- Test: `tests/test_gate.py`

**Interfaces:**
- Produces:
  - `decode(jpeg: bytes) -> np.ndarray` — RGB uint8 HxWx3.
  - `board(a: np.ndarray, b: np.ndarray) -> np.ndarray` — 8×8 float board.
  - `verdict(mask: np.ndarray) -> str` — `"light" | "change" | "skip"`.
  - `class Gate` with `observe(frame: np.ndarray) -> GateResult`.
  - `@dataclass GateResult: verdict: str; streak: int; send: bool` where `verdict in ("first","skip","change","light")`.

Straight port of `camera_v3.ipynb` cells 6 and 10. `Gate` carries `anchor`, `streak`, `prev` — the loop variables from cell 10.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_gate.py
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.gate import GRID, PERSIST, THRESHOLD, Gate, board, decode, verdict

DATA = Path(__file__).resolve().parents[1] / "data"


def frame(name: str) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.imread(str(DATA / name)), cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (1280, 720), interpolation=cv2.INTER_AREA)


@pytest.fixture
def base() -> np.ndarray:
    return (30 + frame("1.png").astype(np.float32) * 0.7).astype(np.uint8)  # headroom for +20


def test_board_shape_and_same_frame_is_zero(base):
    b = board(base, base)
    assert b.shape == (GRID, GRID)
    assert b.max() == 0


def test_lamp_is_below_threshold_and_object_above(base):
    lamp = np.clip(base.astype(np.int16) + 20, 0, 255).astype(np.uint8)
    covered = base.copy()
    covered[:90, :160] = 0
    assert board(base, lamp).max() < THRESHOLD < board(base, covered).max()


def test_recolour_same_luma_is_seen(base):
    grey, tinted = base.copy(), base.copy()
    grey[:90, :160], tinted[:90, :160] = 128, (200, 96, 128)
    assert board(grey, tinted).max() > THRESHOLD


def test_verdicts(base):
    covered = base.copy()
    covered[:90, :160] = 0
    assert verdict(board(base, base) > THRESHOLD) == "skip"
    assert verdict(board(base, covered) > THRESHOLD) == "change"
    assert verdict(np.ones((GRID, GRID), dtype=bool)) == "light"


def test_gate_sends_first_then_needs_persist(base):
    covered = base.copy()
    covered[:90, :160] = 0
    gate = Gate()
    first = gate.observe(base)
    assert (first.verdict, first.send) == ("first", True)
    assert gate.observe(base).send is False
    one = gate.observe(covered)
    assert (one.verdict, one.streak, one.send) == ("change", 1, False)
    two = gate.observe(covered)
    assert (two.verdict, two.streak, two.send) == ("change", PERSIST, True)
    # the anchor moved: the same picture is now quiet
    assert gate.observe(covered).verdict == "skip"


def test_passing_hand_does_not_send(base):
    left, right = base.copy(), base.copy()
    left[:90, :160] = 0
    right[:90, -160:] = 0
    gate = Gate()
    gate.observe(base)
    assert gate.observe(left).streak == 1
    assert gate.observe(right).streak == 1  # different tiles: a new change, not the same one holding


def test_decode_roundtrip(base):
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(base, cv2.COLOR_RGB2BGR))
    assert ok
    decoded = decode(buffer.tobytes())
    assert decoded.shape == base.shape
    assert board(base, decoded).max() < THRESHOLD
```

- [ ] **Step 2: Run to verify they fail**

Run: `poetry run pytest tests/test_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.gate'`

- [ ] **Step 3: Implement**

```python
# src/gate.py
"""Which frames are worth a model call. Port of notebooks/camera_v3.ipynb, cells 6 and 10."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

GATE_SIZE = (128, 72)  # width, height; 16x9 px per tile
GRID = 8  # 8x8 board of tiles
THRESHOLD = 6.6  # Lab levels in the loudest tile; calibrated on the sample footage
GLOBAL = 0.5  # loud tiles over this share of the board: light or exposure, not an object
PERSIST = 2  # frames in a row a change must hold before it is worth sending
OVERLAP = 0.5  # share of loud tiles that must repeat from the previous frame


def decode(jpeg: bytes) -> np.ndarray:
    """JPEG bytes -> RGB uint8 array."""
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("not a decodable image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def prep(x: np.ndarray) -> np.ndarray:
    """Small, blurred Lab. L is centred: a lamp moves the mean, not the room."""
    x = cv2.resize(x, GATE_SIZE, interpolation=cv2.INTER_AREA)
    x = cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_RGB2LAB), (0, 0), 2).astype(np.float32)
    x[..., 0] -= x[..., 0].mean()
    return x


def board(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """GRID x GRID board: mean of the loudest channel difference per tile."""
    diff = np.abs(prep(a) - prep(b)).max(axis=2)
    h, w = diff.shape
    return diff.reshape(GRID, h // GRID, GRID, w // GRID).mean(axis=(1, 3))


def verdict(mask: np.ndarray) -> str:
    """mask = board > THRESHOLD. 'skip' | 'change' | 'light'; only 'change' can lead to a call."""
    return "light" if mask.mean() > GLOBAL else "change" if mask.any() else "skip"


@dataclass
class GateResult:
    verdict: str  # first | skip | change | light
    streak: int
    send: bool


class Gate:
    """Per-session gate state: the anchor and how long the current change has held."""

    def __init__(self) -> None:
        self.anchor: Optional[np.ndarray] = None
        self.streak = 0
        self.prev: Optional[np.ndarray] = None

    def observe(self, frame: np.ndarray) -> GateResult:
        if self.anchor is None:
            self.anchor = frame
            return GateResult("first", 0, True)
        mask = board(self.anchor, frame) > THRESHOLD
        v = verdict(mask)
        if v == "skip":
            self.streak, self.prev = 0, None
            return GateResult(v, 0, False)
        same = self.prev is not None and (mask & self.prev).sum() / mask.sum() > OVERLAP
        self.streak = self.streak + 1 if same else 1
        self.prev = mask
        result = GateResult(v, self.streak, self.streak >= PERSIST and v == "change")
        if self.streak >= PERSIST:  # change confirmed or light settled: new baseline
            self.anchor, self.streak, self.prev = frame, 0, None
        return result
```

- [ ] **Step 4: Run tests**

Run: `poetry run pytest tests/test_gate.py -v`
Expected: 7 passed. If `test_lamp_is_below_threshold_and_object_above` fails on the `lamp` side, the `+20` overflowed — the fixture's `0.7` scaling is there to prevent that; check it was applied.

- [ ] **Step 5: Format, lint, commit**

```bash
make format && make lint && make test
git add src/gate.py tests/test_gate.py
git commit -m "feat: frame gate ported from camera_v3 notebook"
```

---

### Task 4: Tracker (edge logic)

**Files:**
- Create: `src/tracker.py`
- Test: `tests/test_tracker.py`

**Interfaces:**
- Produces: `class Tracker(direction: str, persist: int = 2)` with `state: bool | None` and `update(state_now: bool) -> bool` (True = event fired on this update).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tracker.py
from src.tracker import Tracker


def feed(tracker: Tracker, answers: str) -> list[bool]:
    """'ttff' -> updates; returns which ones fired."""
    return [tracker.update(c == "t") for c in answers]


def test_baseline_never_fires():
    assert feed(Tracker("rising"), "tt") == [False, False]
    assert feed(Tracker("falling"), "ff") == [False, False]


def test_rising_fires_once_on_confirmed_flip():
    t = Tracker("rising")
    assert feed(t, "ff" + "tt" + "tttt") == [False, False, False, True, False, False, False, False]
    assert t.state is True


def test_falling_fires_on_true_to_false():
    assert feed(Tracker("falling"), "tt" + "ff") == [False, False, False, True]


def test_rising_ignores_true_to_false():
    assert feed(Tracker("rising"), "tt" + "ff") == [False, False, False, False]


def test_flicker_does_not_flip():
    t = Tracker("rising")
    assert feed(t, "ff" + "t" + "f" + "t" + "f") == [False] * 6
    assert t.state is False


def test_fires_again_after_going_back_down():
    fired = feed(Tracker("rising"), "ff" + "tt" + "ff" + "tt")
    assert fired == [False, False, False, True, False, False, False, True]


def test_persist_one_flips_immediately():
    assert feed(Tracker("rising", persist=1), "ft") == [False, True]
```

- [ ] **Step 2: Run to verify they fail**

Run: `poetry run pytest tests/test_tracker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tracker'`

- [ ] **Step 3: Implement**

```python
# src/tracker.py
"""The event: the confirmed predicate state flips in the direction the user asked for."""

from typing import Optional


class Tracker:
    def __init__(self, direction: str, persist: int = 2) -> None:
        assert direction in ("rising", "falling"), direction
        self.direction = direction
        self.persist = persist
        self.state: Optional[bool] = None  # confirmed state; None until the baseline
        self._candidate: Optional[bool] = None
        self._streak = 0

    def update(self, state_now: bool) -> bool:
        """Feed one model answer. Returns True when the event fires on this answer."""
        if state_now == self.state:
            self._candidate, self._streak = None, 0
            return False
        if state_now == self._candidate:
            self._streak += 1
        else:
            self._candidate, self._streak = state_now, 1
        if self._streak < self.persist:
            return False
        previous, self.state = self.state, state_now
        self._candidate, self._streak = None, 0
        return previous is not None and state_now == (self.direction == "rising")
```

- [ ] **Step 4: Run tests**

Run: `poetry run pytest tests/test_tracker.py -v`
Expected: 7 passed.

- [ ] **Step 5: Format, lint, commit**

```bash
make format && make lint && make test
git add src/tracker.py tests/test_tracker.py
git commit -m "feat: event tracker with persist and direction"
```

---

### Task 5: Perception (xAI Grok)

**Files:**
- Create: `src/perception.py`
- Test: `tests/test_perception.py`

**Interfaces:**
- Consumes: `config.XAI_API_KEYS`, `config.XAI_MODEL`; the prompt and payload from `scripts/grok_check.py`.
- Produces:
  - `@dataclass Rule: predicate: str; direction: str; is_transition: bool`
  - `@dataclass Observation: state: bool; evidence: str`
  - `class Perception(Protocol)`: `async def normalize(self, rule: str) -> Rule`; `async def detect(self, jpeg: bytes, predicate: str) -> Observation`
  - `class GrokPerception(keys: list[str], model: str)` implementing it; `async def aclose()`.
  - `class PerceptionError(Exception)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_perception.py
import json

import httpx
import pytest

from src.perception import GrokPerception, PerceptionError


def reply(content: str, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json={"choices": [{"message": {"content": content}}]})


def make(handler) -> GrokPerception:
    p = GrokPerception(keys=["A", "B"], model="m")
    p.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.x.ai/v1")
    return p


@pytest.mark.asyncio
async def test_detect_parses_json():
    async def handler(request):
        assert request.headers["authorization"] == "Bearer A"
        body = json.loads(request.content)
        assert body["model"] == "m"
        assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        return reply('{"state_now": true, "evidence": "cat on table"}')

    obs = await make(handler).detect(b"jpegbytes", "a cat is on the table")
    assert (obs.state, obs.evidence) == (True, "cat on table")


@pytest.mark.asyncio
async def test_detect_regex_net_when_json_is_off():
    async def handler(request):
        return reply("Sure! {state_now: false} the table is empty")

    obs = await make(handler).detect(b"x", "p")
    assert obs.state is False


@pytest.mark.asyncio
async def test_keys_round_robin_and_retry_on_429():
    seen = []

    async def handler(request):
        seen.append(request.headers["authorization"][-1])
        if len(seen) == 2:
            return httpx.Response(429, json={"error": "slow down"})
        return reply('{"state_now": false, "evidence": ""}')

    p = make(handler)
    await p.detect(b"x", "p")  # A
    await p.detect(b"x", "p")  # B -> 429 -> retry on A
    assert seen == ["A", "B", "A"]


@pytest.mark.asyncio
async def test_error_after_retry_exhausted():
    async def handler(request):
        return httpx.Response(500, text="boom")

    with pytest.raises(PerceptionError):
        await make(handler).detect(b"x", "p")


@pytest.mark.asyncio
async def test_normalize_rule():
    async def handler(request):
        body = json.loads(request.content)
        assert "cat jumps onto the table" in body["messages"][0]["content"]
        return reply('{"predicate": "a cat is on the table", "direction": "rising", "is_transition": true}')

    rule = await make(handler).normalize("the cat jumps onto the table")
    assert (rule.predicate, rule.direction, rule.is_transition) == ("a cat is on the table", "rising", True)


@pytest.mark.asyncio
async def test_normalize_bad_direction_is_error():
    async def handler(request):
        return reply('{"predicate": "x", "direction": "sideways", "is_transition": true}')

    with pytest.raises(PerceptionError):
        await make(handler).normalize("x")
```

`pytest-asyncio` is needed: `poetry add --group dev pytest-asyncio` and add to `pyproject.toml` under `[tool.pytest.ini_options]` the line `asyncio_mode = "auto"` — this section is pytest config, not a dependency, so editing it by hand is fine. With `asyncio_mode = "auto"` the `@pytest.mark.asyncio` decorators are optional; keep them for clarity.

- [ ] **Step 2: Run to verify they fail**

Run: `poetry run pytest tests/test_perception.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.perception'`

- [ ] **Step 3: Implement**

```python
# src/perception.py
"""One question per call to a vision model. Port of notebooks/groq.ipynb, pointed at xAI."""

import base64
import json
import re
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Iterator, Protocol

import httpx
from loguru import logger

BASE_URL = "https://api.x.ai/v1"

DETECT_PROMPT = (
    "You look at a single still frame from a fixed security camera.\n"
    "Answer only this about THIS frame: is the following true right now?\n"
    "  {predicate}\n"
    "If the frame is too dark or unclear to tell, answer false.\n"
    'Reply with JSON only: {{"state_now": true|false, "evidence": "<a few words on what you see>"}}'
)

NORMALIZE_PROMPT = (
    "A user wants a camera to notify them when something HAPPENS. Their words:\n"
    "  {rule}\n"
    "Rewrite it as a STATE that is either true or false in a single still frame, plus the\n"
    "direction of the change the user is waiting for.\n"
    "- predicate: a short present-tense sentence about the scene, e.g. 'a cat is on the table'\n"
    "- direction: 'rising' if the user waits for the predicate to become true (appears, arrives,\n"
    "  jumps on, turns on), 'falling' if they wait for it to become false (leaves, goes away, turns off)\n"
    "- is_transition: false if the words describe no change at all (just an object or a scene)\n"
    'Reply with JSON only: {{"predicate": "...", "direction": "rising"|"falling", "is_transition": true|false}}'
)


class PerceptionError(Exception):
    pass


@dataclass
class Rule:
    predicate: str
    direction: str
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
        self._retries = min(len(keys), 2)  # one retry on the next key, if there is one
        self.client = httpx.AsyncClient(base_url=BASE_URL, timeout=30)

    async def aclose(self) -> None:
        await self.client.aclose()

    async def _ask(self, content: Any) -> str:
        payload = {"model": self.model, "temperature": 0, "max_tokens": 96,
                   "messages": [{"role": "user", "content": content}]}
        last: Exception | None = None
        for _ in range(self._retries):
            key = next(self._keys)
            try:
                response = await self.client.post("/chat/completions", json=payload,
                                                  headers={"Authorization": f"Bearer {key}"})
                if response.status_code in (429, 500, 502, 503):
                    last = PerceptionError(f"xai {response.status_code}: {response.text[:120]}")
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
        return Rule(str(data["predicate"]), direction, bool(data.get("is_transition", True)))

    async def detect(self, jpeg: bytes, predicate: str) -> Observation:
        image = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        raw = await self._ask([
            {"type": "text", "text": DETECT_PROMPT.format(predicate=predicate)},
            {"type": "image_url", "image_url": {"url": image}},
        ])
        data = _json(raw)
        if "state_now" in data:
            return Observation(bool(data["state_now"]), str(data.get("evidence", "")))
        found = re.findall(r"true|false", raw.lower())  # json_object guarantees JSON, not the schema
        if not found:
            raise PerceptionError(f"no verdict in: {raw[:120]}")
        return Observation(found[-1] == "true", "<unparsed>")
```

- [ ] **Step 4: Run tests**

Run: `poetry run pytest tests/test_perception.py -v`
Expected: 6 passed.

- [ ] **Step 5: Format, lint, commit**

```bash
make format && make lint && make test
git add src/perception.py tests/test_perception.py pyproject.toml poetry.lock
git commit -m "feat: grok perception with key rotation"
```

---

### Task 6: Sessions and notifier

**Files:**
- Create: `src/session.py`, `src/notifier.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Consumes: `Gate`, `Tracker`, `Rule`.
- Produces:
  - `@dataclass Event: n: int; at: str; text: str; image: bytes`
  - `@dataclass Session: id: str; rule: str; predicate: str; direction: str; gate: Gate; tracker: Tracker; evidence: str = ""; busy: bool = False; fired: bool = False; last_seen: float; events: list[Event]; lock: asyncio.Lock; task: Optional[asyncio.Task]`
  - `class SessionFull(Exception)`
  - `class SessionStore(max_sessions: int, ttl: float)`: `create(rule: str, spec: Rule) -> Session`, `get(session_id) -> Session` (raises `KeyError`), `touch(session)`, `delete(session_id)`, `sweep(now: float | None = None) -> int`, `__len__`.
  - `class Notifier` with `async def notify(self, session: Session, event: Event) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_session.py
import pytest

from src.perception import Rule
from src.session import SessionFull, SessionStore

RULE = Rule("a cat is on the table", "rising", True)


def test_create_and_get():
    store = SessionStore(max_sessions=2, ttl=30)
    s = store.create("the cat jumps on the table", RULE)
    assert store.get(s.id) is s
    assert (s.predicate, s.direction, s.tracker.direction) == (RULE.predicate, "rising", "rising")
    assert len(store) == 1


def test_cap():
    store = SessionStore(max_sessions=2, ttl=30)
    store.create("a", RULE)
    store.create("b", RULE)
    with pytest.raises(SessionFull):
        store.create("c", RULE)


def test_ttl_frees_slot():
    store = SessionStore(max_sessions=1, ttl=30)
    s = store.create("a", RULE)
    s.last_seen = 0.0
    assert store.sweep(now=31.0) == 1
    with pytest.raises(KeyError):
        store.get(s.id)
    store.create("b", RULE)  # slot is free again


def test_create_sweeps_first():
    store = SessionStore(max_sessions=1, ttl=30)
    s = store.create("a", RULE)
    s.last_seen = 0.0
    store.create("b", RULE)  # would raise SessionFull without the sweep


def test_delete():
    store = SessionStore(max_sessions=1, ttl=30)
    s = store.create("a", RULE)
    store.delete(s.id)
    assert len(store) == 0
    store.delete("missing")  # no error
```

- [ ] **Step 2: Run to verify they fail**

Run: `poetry run pytest tests/test_session.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.session'`

- [ ] **Step 3: Implement**

```python
# src/session.py
"""In-memory sessions: one gate + one tracker per browser tab, capped, expiring on silence."""

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional

from src.gate import Gate
from src.perception import Rule
from src.tracker import Tracker


class SessionFull(Exception):
    pass


@dataclass
class Event:
    n: int
    at: str  # ISO-8601 UTC
    text: str
    image: bytes


@dataclass
class Session:
    id: str
    rule: str
    predicate: str
    direction: str
    gate: Gate
    tracker: Tracker
    evidence: str = ""
    busy: bool = False  # a model call is in flight
    fired: bool = False  # an event fired, not yet reported in a FrameStatus
    last_seen: float = field(default_factory=time.monotonic)
    events: list[Event] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # frames of one tab, in order
    task: Optional["asyncio.Task[None]"] = None  # keeps the background model call alive


class SessionStore:
    def __init__(self, max_sessions: int, ttl: float) -> None:
        self.max_sessions = max_sessions
        self.ttl = ttl
        self._sessions: dict[str, Session] = {}

    def __len__(self) -> int:
        return len(self._sessions)

    def create(self, rule: str, spec: Rule) -> Session:
        self.sweep()
        if len(self._sessions) >= self.max_sessions:
            raise SessionFull()
        session = Session(secrets.token_urlsafe(6), rule, spec.predicate, spec.direction,
                          Gate(), Tracker(spec.direction))
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session:
        return self._sessions[session_id]

    def touch(self, session: Session) -> None:
        session.last_seen = time.monotonic()

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def sweep(self, now: float | None = None) -> int:
        now = time.monotonic() if now is None else now
        dead = [k for k, s in self._sessions.items() if now - s.last_seen > self.ttl]
        for k in dead:
            del self._sessions[k]
        return len(dead)
```

```python
# src/notifier.py
"""Where events go. Logs only until the Telegram phase."""

from loguru import logger

from src.session import Event, Session


class Notifier:
    async def notify(self, session: Session, event: Event) -> None:
        # ponytail: Telegram lands here — photo + caption to the chat bound to session.id
        logger.info("EVENT session={} n={} {} ({} bytes)", session.id, event.n, event.text, len(event.image))
```

- [ ] **Step 4: Run tests**

Run: `poetry run pytest tests/test_session.py -v`
Expected: 5 passed.

- [ ] **Step 5: Format, lint, commit**

```bash
make format && make lint && make test
git add src/session.py src/notifier.py tests/test_session.py
git commit -m "feat: in-memory session store and notifier stub"
```

---

### Task 7: Engine and FastAPI app

**Files:**
- Create: `src/engine.py`, `src/app.py`, `static/index.html` (placeholder only — the client plan replaces it)
- Modify: `main.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `async def handle_frame(session: Session, jpeg: bytes, perception: Perception, notifier: Notifier) -> dict` — returns the spec's `FrameStatus`.
  - `def create_app(perception: Perception, store: SessionStore, notifier: Notifier) -> FastAPI` — routes per the spec's HTTP contract.
  - `app.py` also exposes `app = create_app(...)` built from config for uvicorn.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_app.py
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.app import create_app
from src.notifier import Notifier
from src.perception import Observation, Rule
from src.session import SessionStore

DATA = Path(__file__).resolve().parents[1] / "data"


class FakePerception:
    def __init__(self) -> None:
        self.answers: list[bool] = []
        self.calls = 0

    async def normalize(self, rule: str) -> Rule:
        if rule == "a cat":
            return Rule("a cat is visible", "rising", False)
        return Rule("a cat is on the table", "falling" if "leaves" in rule else "rising", True)

    async def detect(self, jpeg: bytes, predicate: str) -> Observation:
        self.calls += 1
        return Observation(self.answers.pop(0), "fake")


class SpyNotifier(Notifier):
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def notify(self, session, event) -> None:
        self.sent.append(event.text)


def jpeg(name: str, black_tile: bool = False) -> bytes:
    bgr = cv2.resize(cv2.imread(str(DATA / name)), (1280, 720))
    bgr = (30 + bgr.astype(np.float32) * 0.7).astype(np.uint8)
    if black_tile:
        bgr[:90, :160] = 0
    return cv2.imencode(".jpg", bgr)[1].tobytes()


@pytest.fixture
def world():
    perception, notifier = FakePerception(), SpyNotifier()
    app = create_app(perception, SessionStore(max_sessions=2, ttl=30), notifier)
    return TestClient(app), perception, notifier


def post_frame(client, sid, data):
    return client.post(f"/session/{sid}/frame", files={"frame": ("f.jpg", data, "image/jpeg")})


def test_health_and_index(world):
    client, _, _ = world
    assert client.get("/health").json() == {"ok": True, "sessions": 0}
    assert client.get("/").status_code == 200


def test_create_session_echoes_reading(world):
    client, _, _ = world
    r = client.post("/session", json={"rule": "the cat jumps onto the table"})
    assert r.status_code == 201
    assert r.json()["predicate"] == "a cat is on the table"
    assert r.json()["direction"] == "rising"


def test_non_transition_refused(world):
    client, _, _ = world
    r = client.post("/session", json={"rule": "a cat"})
    assert r.status_code == 400
    assert r.json()["error"] == "not_a_transition"


def test_full(world):
    client, _, _ = world
    for _ in range(2):
        assert client.post("/session", json={"rule": "x happens"}).status_code == 201
    r = client.post("/session", json={"rule": "x happens"})
    assert (r.status_code, r.json()["error"]) == (503, "full")


def test_frame_flow_fires_once(world):
    client, perception, notifier = world
    sid = client.post("/session", json={"rule": "the cat jumps onto the table"}).json()["session_id"]
    perception.answers = [False, False, True, True, True]
    quiet, changed = jpeg("1.png"), jpeg("1.png", black_tile=True)

    # The model answers in a background task, so each answer shows on the NEXT status.
    s = post_frame(client, sid, quiet).json()  # first frame: sent, baseline candidate
    assert (s["gate"], s["sent"], s["state"]) == ("first", True, None)
    s = post_frame(client, sid, quiet).json()  # same picture: skipped; False 1/2 -> still None
    assert (s["gate"], s["sent"], s["state"]) == ("skip", False, None)
    assert post_frame(client, sid, changed).json()["streak"] == 1  # change, not yet persisted
    assert post_frame(client, sid, changed).json()["sent"] is True  # persisted: sent -> False 2/2
    # anchor moved; back to the quiet picture: baseline False is confirmed now
    s = post_frame(client, sid, quiet).json()
    assert (s["streak"], s["state"]) == (1, False)
    assert post_frame(client, sid, quiet).json()["sent"] is True  # -> True 1/2
    s = post_frame(client, sid, changed).json()
    assert (s["streak"], s["fired"], s["state"]) == (1, False, False)
    assert post_frame(client, sid, changed).json()["sent"] is True  # -> True 2/2: fires
    s = post_frame(client, sid, quiet).json()
    assert (s["fired"], s["state"], s["events"]) == (True, True, 1)
    assert post_frame(client, sid, quiet).json()["fired"] is False  # reported once
    assert notifier.sent == ["a cat is on the table — became true"]
    assert perception.calls == 4

    view = client.get(f"/session/{sid}").json()
    assert view["events"][0]["n"] == 0 and view["state"] is True
    img = client.get(f"/session/{sid}/events/0.jpg")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    ev = client.get(f"/session/{sid}/events/0").json()
    assert ev["n"] == 0 and ev["text"] == view["events"][0]["text"]
    assert ev["image"].startswith("data:image/jpeg;base64,")
    assert client.get(f"/session/{sid}/events/1").status_code == 404


def test_unknown_session_404(world):
    client, _, _ = world
    assert post_frame(client, "nope", jpeg("1.png")).status_code == 404
    assert client.get("/session/nope").status_code == 404


def test_bad_image_400(world):
    client, _, _ = world
    sid = client.post("/session", json={"rule": "x happens"}).json()["session_id"]
    assert post_frame(client, sid, b"not a jpeg").status_code == 400


def test_delete(world):
    client, _, _ = world
    sid = client.post("/session", json={"rule": "x happens"}).json()["session_id"]
    assert client.delete(f"/session/{sid}").status_code == 204
    assert client.get(f"/session/{sid}").status_code == 404
```

Note on `test_frame_flow_fires_once`: the fake answers are consumed in order — `False` (first frame), `False` (confirms baseline false), `True`, `True` (second true → flip → fire). The 5th answer is spare. `perception.calls == 4` pins that dropped/skipped frames never reach the model. The fake `detect` never awaits anything, so the background task finishes on the next loop tick, before `TestClient` sends the following request — that is why the answer is asserted on the next frame's status, never on the frame that was sent.

- [ ] **Step 2: Run to verify they fail**

Run: `poetry run pytest tests/test_app.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.app'`

- [ ] **Step 3: Implement the engine**

```python
# src/engine.py
"""One frame through the loop: gate -> model -> tracker -> notifier.

Multi-user: many browser tabs post frames into one event loop. Two rules keep one user
from stalling the others: decode + gate (cv2/numpy, CPU-bound) run in a worker thread,
and the model call runs as a background task — POST /frame returns at once with
sent=True, and the answer shows up in the next frame's status (state/evidence/fired/events).
"""

import asyncio
from datetime import datetime, timezone

from loguru import logger

from src.gate import GateResult, decode
from src.notifier import Notifier
from src.perception import Perception, PerceptionError
from src.session import Event, Session


def _status(session: Session, gate: GateResult, sent: bool) -> dict:
    fired, session.fired = session.fired, False  # reported once, on the next status
    return {"gate": gate.verdict, "streak": gate.streak, "sent": sent, "busy": session.busy,
            "state": session.tracker.state, "evidence": session.evidence, "fired": fired,
            "events": len(session.events)}


async def handle_frame(session: Session, jpeg: bytes, perception: Perception, notifier: Notifier) -> dict:
    async with session.lock:  # gate state is per-session and not thread-safe
        frame = await asyncio.to_thread(decode, jpeg)  # raises ValueError on junk
        gate = await asyncio.to_thread(session.gate.observe, frame)
        if not gate.send or session.busy:
            return _status(session, gate, sent=False)
        session.busy = True
        session.task = asyncio.create_task(perceive(session, jpeg, perception, notifier))
        return _status(session, gate, sent=True)


async def perceive(session: Session, jpeg: bytes, perception: Perception, notifier: Notifier) -> None:
    """Background: ask the model, update the tracker, fire the edge. Never raises."""
    try:
        observation = await perception.detect(jpeg, session.predicate)
    except PerceptionError as e:
        logger.warning("session={} perception failed: {}", session.id, e)
        return
    finally:
        session.busy = False
    session.evidence = observation.evidence
    if session.tracker.update(observation.state):
        became = "true" if session.direction == "rising" else "false"
        event = Event(len(session.events), datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      f"{session.predicate} — became {became}", jpeg)
        session.events.append(event)
        session.fired = True
        await notifier.notify(session, event)
```

- [ ] **Step 4: Implement the app**

```python
# src/app.py
"""HTTP surface. The contract lives in docs/superpowers/specs/2026-09-12-camera-events-design.md."""

import base64
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import config
from src.engine import handle_frame
from src.notifier import Notifier
from src.perception import GrokPerception, Perception, PerceptionError
from src.session import SessionFull, SessionStore

STATIC = Path(__file__).resolve().parents[1] / "static"


class NewSession(BaseModel):
    rule: str


def create_app(perception: Perception, store: SessionStore, notifier: Notifier) -> FastAPI:
    app = FastAPI(title="camera events")

    def session_or_404(session_id: str):
        try:
            return store.get(session_id)
        except KeyError:
            raise HTTPException(404, "no such session")

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    async def health():
        return {"ok": True, "sessions": len(store)}

    @app.post("/session", status_code=201)
    async def create_session(body: NewSession):
        rule = body.rule.strip()
        if not rule:
            return JSONResponse({"error": "empty_rule", "hint": "describe something that happens"}, status_code=400)
        try:
            spec = await perception.normalize(rule)
        except PerceptionError as e:
            return JSONResponse({"error": "perception", "hint": str(e)}, status_code=502)
        if not spec.is_transition:
            return JSONResponse({"error": "not_a_transition",
                                 "hint": "describe something that happens — e.g. 'the cat jumps onto the table'"},
                                status_code=400)
        try:
            session = store.create(rule, spec)
        except SessionFull:
            return JSONResponse({"error": "full"}, status_code=503)
        return {"session_id": session.id, "predicate": session.predicate, "direction": session.direction}

    @app.post("/session/{session_id}/frame")
    async def frame(session_id: str, frame: UploadFile = File(...)):
        session = session_or_404(session_id)
        store.touch(session)
        try:
            return await handle_frame(session, await frame.read(), perception, notifier)
        except ValueError:
            raise HTTPException(400, "not a decodable image")

    @app.get("/session/{session_id}")
    async def view(session_id: str):
        s = session_or_404(session_id)
        return {"session_id": s.id, "rule": s.rule, "predicate": s.predicate, "direction": s.direction,
                "state": s.tracker.state, "evidence": s.evidence,
                "events": [{"n": e.n, "at": e.at, "text": e.text} for e in s.events]}

    @app.get("/session/{session_id}/events/{n}")
    async def event(session_id: str, n: int):
        s = session_or_404(session_id)
        if n >= len(s.events):
            raise HTTPException(404, "no such event")
        e = s.events[n]
        return {"n": e.n, "at": e.at, "text": e.text,
                "image": "data:image/jpeg;base64," + base64.b64encode(e.image).decode()}

    @app.get("/session/{session_id}/events/{n}.jpg")
    async def event_image(session_id: str, n: int):
        s = session_or_404(session_id)
        if n >= len(s.events):
            raise HTTPException(404, "no such event")
        return Response(s.events[n].image, media_type="image/jpeg")

    @app.delete("/session/{session_id}", status_code=204)
    async def delete(session_id: str):
        store.delete(session_id)
        return Response(status_code=204)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def default_app() -> FastAPI:
    perception = GrokPerception(config.XAI_API_KEYS, config.XAI_MODEL)
    return create_app(perception, SessionStore(config.MAX_SESSIONS, config.SESSION_TTL), Notifier())
```

Placeholder page so `GET /` works before the client plan lands:

```html
<!-- static/index.html -->
<!doctype html><meta charset="utf-8"><title>camera events</title>
<p>server is up — the client page is built in plans/2026-09-12-web-client.md</p>
```

- [ ] **Step 5: Wire `main.py`**

```python
"""Точка входа."""

import sys

import uvicorn
from loguru import logger

from src import config


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level=config.LOG_LEVEL)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("starting on :{} model={} keys={}", config.PORT, config.XAI_MODEL, len(config.XAI_API_KEYS))
    uvicorn.run("src.app:default_app", factory=True, host="0.0.0.0", port=config.PORT, log_level="warning")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests**

Run: `poetry run pytest tests/test_app.py -v`
Expected: 8 passed. If `test_frame_flow_fires_once` fails at the first `changed` frame with `streak == 2`, the two consecutive `changed` uploads are identical bytes and `OVERLAP` is satisfied — that is intended; check the earlier `quiet` frame was actually posted first.

- [ ] **Step 7: Smoke it locally**

```bash
make dev   # in one terminal
curl -s localhost:8000/health
curl -s -X POST localhost:8000/session -H 'content-type: application/json' -d '{"rule":"the cat jumps onto the table"}'
```

Expected: `{"ok":true,"sessions":0}` and a `201` with the normalized predicate from the real Grok. Then, with the `session_id` printed:

```bash
curl -s -F frame=@data/1.png localhost:8000/session/<id>/frame
```

Expected: `{"gate":"first","sent":true,...}` — PNG decodes fine through `cv2.imdecode` too.

- [ ] **Step 8: Format, lint, commit**

```bash
make format && make lint && make test
git add src/engine.py src/app.py static/index.html main.py tests/test_app.py
git commit -m "feat: engine loop and http api"
```

---

### Task 8: Render deployment

**Files:**
- Modify: `render.yaml`

- [ ] **Step 1: Declare env and health check**

Replace `render.yaml` with:

```yaml
services:
  - type: web
    name: codex-hackathon
    runtime: python
    plan: free
    buildCommand: pip install poetry && poetry install --only main --no-root
    startCommand: poetry run python main.py
    healthCheckPath: /health
    envVars:
      - key: PYTHON_VERSION
        value: "3.10"
      - key: POETRY_VIRTUALENVS_IN_PROJECT
        value: "true"
      - key: LOG_LEVEL
        value: INFO
      - key: XAI_API_KEYS
        sync: false
      - key: XAI_MODEL
        value: grok-4.6
      - key: MAX_SESSIONS
        value: "10"
      - key: SESSION_TTL
        value: "30"
```

`sync: false` makes Render prompt for the value in the dashboard; the key never enters git. Render injects `PORT` itself. If Task 2 picked a different model, put that id here.

- [ ] **Step 2: Commit and push**

```bash
git add render.yaml
git commit -m "chore: render env and health check"
git push origin master
```

- [ ] **Step 3: Verify**

In the Render dashboard: set `XAI_API_KEYS`, wait for the deploy, then:

```bash
curl -s https://<service>.onrender.com/health
```

Expected: `{"ok":true,"sessions":0}`. Open `https://<service>.onrender.com/` on a phone: the placeholder page loads over HTTPS (the client plan's page replaces it and the camera prompt appears).

Known ceiling: free plan sleeps after 15 min idle and the first request takes ~30 s to wake. Hit `/health` before a demo.

---

## Self-review

- **Spec coverage:** normalization + refusal (T5, T7), gate verbatim (T3), perception single-frame JSON with key rotation (T5), persist-2 edge with direction and baseline (T4), one in-flight call per session (T7 `busy`), notifier stub (T6), cap 10 + TTL 30 (T6), full HTTP contract incl. event images and delete (T7), config + `.env.example` (T1), Render with TLS (T8), Grok check first (T2). No cooldown/heartbeat by decision.
- **Types:** `Perception.detect(jpeg: bytes, predicate: str) -> Observation` is used identically in T5, T7 engine and T7 fake. `SessionStore.create(rule: str, spec: Rule)` matches T6 and T7. `GateResult.verdict` values match the spec's `gate` enum. `Event(n, at, text, image)` matches T6, T7 and the `SessionView` shape.
- **Client seam:** field names in `FrameStatus` and `SessionView` match the spec table verbatim.
