# Camera Events — Design

Public web page opens the device camera. User writes one rule in plain language describing
a *change* ("the cat jumps onto the table", "someone leaves the room"). The server watches
the frames and, the moment the change happens, notifies the user with text plus the frame
that proves it.

Settled in the design session on 2026-09-12. Two parallel plans implement it:
`plans/2026-09-12-server-engine.md` and `plans/2026-09-12-web-client.md`.

## Topology

Two components over HTTPS. No persistence.

- **Client** — one static page, vanilla JS. `getUserMedia` → `<canvas>` → JPEG →
  `POST /session/{id}/frame` on a timer. Served by the server itself.
- **Server** — FastAPI + uvicorn, async. Gate, perception, sessions, notifier. Hosted on
  Render (`render.yaml`, auto-deploy on push to `master`). Render terminates TLS; browsers
  refuse `getUserMedia` outside `https://` or `localhost`, so TLS is not optional.

## Detection engine

The agent is a loop: sense (gate) → perceive (LLM) → remember (per-session state) → act
(notifier). The LLM answers exactly one question per call; all temporal logic is Python.

1. **Rule normalization** — once, at submit. Text-only LLM call turns the rule into
   `{predicate, direction, is_transition}`. `predicate` is a state visible in one frame
   ("a cat is on the table"); `direction` is `rising` (becomes true) or `falling`
   (becomes false). Rules that describe no change (`is_transition == false`) are refused.
2. **Gate** — ported from `notebooks/camera_v3.ipynb` unchanged: 128×72, Lab diff with
   centred L, Gaussian blur σ=2, 8×8 tiles, compare against the **anchor** (last frame
   sent), `THRESHOLD 6.6`, `GLOBAL 0.5`, `PERSIST 2`, `OVERLAP 0.5`. Verdicts:
   `skip` / `change` / `light`. A `change` held `PERSIST` frames in overlapping tiles →
   frame goes to the model and becomes the anchor. A `light` held `PERSIST` frames → anchor
   moves, nothing sent. The very first frame of a session is sent (baseline state).
3. **Perception** — ported from `notebooks/groq.ipynb`: one frame, one question — is
   `predicate` true in this frame? — JSON `{state_now, evidence}`. Provider: xAI Grok via
   the OpenAI-compatible Chat Completions endpoint, `httpx`. Keys: `XAI_API_KEYS`
   (comma-separated, 1..N), round-robin per call, one retry on the next key on 429/5xx.
4. **Event** — the answer must hold 2 consecutive calls to become the confirmed state.
   Fires once when the confirmed state flips to `true` (rising) or `false` (falling). The
   first confirmed state is a baseline and never fires. No cooldown, no heartbeat.
5. **Concurrency** — one asyncio process serves many browser tabs at once; no user may
   stall another. Per frame: decode + gate (cv2/numpy, CPU-bound) run in a worker thread
   via `asyncio.to_thread` under a per-session lock, so the event loop stays free and one
   session's frames are processed in order. The model call is a background task:
   `POST /frame` returns at once with `sent: true`; the answer lands in session state and
   is reported by the next frame's status. One in-flight LLM call per session. Frames that
   arrive while it runs are gated but not sent (`busy: true`, dropped, not queued). No
   global lock, no queue: `MAX_SESSIONS` tabs × one call each is the whole load.
6. **Notifier** — component with `notify(session_id, text, image_bytes)`. Logs only until
   the Telegram phase (deep-link `/start <code>` binding, photo + caption — later).

## Sessions

In-memory dict. `MAX_SESSIONS = 10`; creating the 11th returns `503 {"error": "full"}`.
A slot is released when no frame has arrived for `SESSION_TTL = 30` s. Rule change =
new session.

## HTTP contract (the seam between the two plans)

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/` | — | `static/index.html` |
| `GET` | `/health` | — | `200 {"ok": true, "sessions": n}` |
| `POST` | `/session` | JSON `{"rule": str}` | `201 {"session_id", "predicate", "direction"}` · `400 {"error": "not_a_transition", "hint": str}` · `503 {"error": "full"}` |
| `POST` | `/session/{id}/frame` | multipart field `frame` (JPEG) | `200 FrameStatus` · `404` |
| `GET` | `/session/{id}` | — | `200 SessionView` · `404` |
| `GET` | `/session/{id}/events/{n}` | — | `200 EventView` (event + proof frame in one call) · `404` |
| `GET` | `/session/{id}/events/{n}.jpg` | — | JPEG · `404` |
| `DELETE` | `/session/{id}` | — | `204` |

```jsonc
// FrameStatus — everything the page needs to render, returned on every frame
{
  "gate": "first" | "skip" | "change" | "light",
  "streak": 0,                 // frames the current verdict has held
  "sent": false,               // this frame went to the model (async; result on a later status)
  "busy": false,               // a model call was already in flight; frame not sent
  "state": true | false | null,// confirmed predicate state, null until the baseline
  "evidence": "cat on the chair",
  "fired": false,              // an event fired since the previous status (model answers in the background)
  "events": 1                  // total events so far
}

// SessionView
{
  "session_id": "k3f9x2",
  "rule": "the cat jumps onto the table",
  "predicate": "a cat is on the table",
  "direction": "rising",
  "state": false,
  "evidence": "cat on the chair",
  "events": [{"n": 0, "at": "2026-09-12T14:32:10Z", "text": "..."}]
}

// EventView
{
  "n": 0,
  "at": "2026-09-12T14:32:10Z",
  "text": "a cat is on the table — became true",
  "image": "data:image/jpeg;base64,..."  // the proof frame, inline
}
```

Frame from client: JPEG, 640 px wide, quality 0.8, every `SAMPLE_MS` (default 1000,
user-adjustable 250–5000).

## Configuration (`src/config.py`, every value has a default, every key in `.env.example`)

| Variable | Default | Meaning |
|---|---|---|
| `XAI_API_KEYS` | `""` | comma-separated xAI keys |
| `XAI_MODEL` | `grok-4.6` | vision model id; the Grok check picks the final one |
| `MAX_SESSIONS` | `10` | parallel session cap |
| `SESSION_TTL` | `30` | seconds without frames before a slot is freed |
| `PORT` | `8000` | Render sets this |
| `LOG_LEVEL`, `DATA_DIR` | existing | |

## Verification

- `tests/test_gate.py` — the synthetic assertions from `camera_v3.ipynb` cell 7 on
  `data/1.png`, and the 7-frame sequence from `frame_gate.ipynb`.
- `tests/test_tracker.py` — edge, persist, direction, baseline.
- `tests/test_session.py` — cap, TTL.
- `tests/test_app.py` — full flow through FastAPI `TestClient` with a fake perception.
- `scripts/grok_check.py` — the `groq.ipynb` 3-frame / 5-case check against xAI; run once
  with real keys before the server work starts, picks `XAI_MODEL`.

## Build order

Server plan: Grok check → config + deps → gate → tracker → sessions → perception → engine +
app → Render. Client plan: mock API → page skeleton → camera + sampling → rule submit →
status + event log → styling. Integration: point the page at the real server.
