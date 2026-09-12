# Web Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One static page that opens the device camera, takes a rule from the user, samples frames to the server on a configurable period, and shows live gate/model status plus an event log with proof thumbnails.

**Architecture:** Three static files (`static/index.html`, `static/app.js`, `static/style.css`) served by the FastAPI server at `/` and `/static/`. No framework, no build step. Built against a 40-line mock of the HTTP contract so it doesn't wait for the server plan; integration is switching the base URL.

**Tech Stack:** HTML, CSS, vanilla JS (`getUserMedia`, `<canvas>`, `fetch`, `FormData`). Mock: FastAPI (installed by the server plan's Task 1 — if it isn't yet, `poetry add fastapi uvicorn python-multipart`).

**Spec:** `docs/superpowers/specs/2026-09-12-camera-events-design.md` — section *HTTP contract* is the whole interface.

## Global Constraints

- The page must work on a phone at ~400 px width and on a laptop; camera opens with `facingMode: "environment"` on phones, falls back to any camera.
- `getUserMedia` needs `https://` or `localhost`. Local dev is `http://localhost:8000`; never test on a LAN IP.
- Frames: JPEG, 640 px wide, quality 0.8, every `SAMPLE_MS` (default 1000, slider 250–5000).
- No external assets: no CDN, no fonts, no icon packs. Everything inline or in `static/`.
- Field names come from the spec's `FrameStatus` / `SessionView` and are not to be renamed.
- Commit messages: no AI attribution lines.
- `.ipynb` files are not touched.

## File structure

| File | Responsibility |
|---|---|
| `scripts/mock_api.py` | the HTTP contract with canned responses, for building the page without the engine |
| `static/index.html` | structure: viewfinder, rule form, status strip, event log |
| `static/app.js` | camera, sampling loop, API calls, rendering |
| `static/style.css` | the look |

---

### Task 1: Mock API

**Files:**
- Create: `scripts/mock_api.py`

**Interfaces:**
- Produces: the spec's routes on `http://localhost:8000`, serving `static/` for `/` and `/static/`. Frame responses cycle through a scripted sequence so every UI state is visible: skip → change (streak 1) → change (sent, state false) → … → fired.

- [ ] **Step 1: Write the mock**

```python
"""Mock of the HTTP contract for building the page. Usage: poetry run python scripts/mock_api.py"""

from itertools import cycle
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

STATIC = Path(__file__).resolve().parents[1] / "static"
app = FastAPI(title="mock camera events")
sessions: dict[str, dict] = {}

SCRIPT = [
    {"gate": "first", "streak": 0, "sent": True, "state": None, "evidence": "", "fired": False},
    {"gate": "skip", "streak": 0, "sent": False, "state": None, "evidence": "", "fired": False},
    {"gate": "skip", "streak": 0, "sent": False, "state": False, "evidence": "empty table", "fired": False},
    {"gate": "change", "streak": 1, "sent": False, "state": False, "evidence": "empty table", "fired": False},
    {"gate": "change", "streak": 2, "sent": True, "state": False, "evidence": "cat on the chair", "fired": False},
    {"gate": "skip", "streak": 0, "sent": False, "state": False, "evidence": "cat on the chair", "fired": False},
    {"gate": "light", "streak": 1, "sent": False, "state": False, "evidence": "cat on the chair", "fired": False},
    {"gate": "change", "streak": 1, "sent": False, "state": False, "evidence": "cat on the chair", "fired": False},
    {"gate": "change", "streak": 2, "sent": True, "state": True, "evidence": "cat on the table", "fired": True},
    {"gate": "skip", "streak": 0, "sent": False, "state": True, "evidence": "cat on the table", "fired": False},
]


class NewSession(BaseModel):
    rule: str


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
async def health():
    return {"ok": True, "sessions": len(sessions)}


@app.post("/session", status_code=201)
async def create(body: NewSession):
    rule = body.rule.strip().lower()
    if rule == "full":
        return JSONResponse({"error": "full"}, status_code=503)
    if not any(w in rule for w in ("jump", "leave", "arrive", "appear", "enter", "go", "happen", "come")):
        return JSONResponse({"error": "not_a_transition",
                             "hint": "describe something that happens — e.g. 'the cat jumps onto the table'"},
                            status_code=400)
    sid = f"mock{len(sessions) + 1}"
    direction = "falling" if "leave" in rule or "go" in rule else "rising"
    sessions[sid] = {"rule": body.rule, "predicate": "a cat is on the table", "direction": direction,
                     "script": cycle(SCRIPT), "events": [], "last": SCRIPT[0], "busy": 0}
    return {"session_id": sid, "predicate": "a cat is on the table", "direction": direction}


@app.post("/session/{sid}/frame")
async def frame(sid: str, frame: UploadFile = File(...)):
    s = sessions.get(sid) or _404()
    data = await frame.read()
    status = dict(next(s["script"]))
    status["busy"] = s["busy"] % 7 == 6  # show the 'busy' state now and then
    s["busy"] += 1
    if status["fired"]:
        s["events"].append({"n": len(s["events"]), "at": "2026-09-12T14:32:10Z",
                            "text": "a cat is on the table — became true", "image": data})
    status["events"] = len(s["events"])
    s["last"] = status
    return status


@app.get("/session/{sid}")
async def view(sid: str):
    s = sessions.get(sid) or _404()
    return {"session_id": sid, "rule": s["rule"], "predicate": s["predicate"], "direction": s["direction"],
            "state": s["last"]["state"], "evidence": s["last"]["evidence"],
            "events": [{k: e[k] for k in ("n", "at", "text")} for e in s["events"]]}


@app.get("/session/{sid}/events/{n}.jpg")
async def image(sid: str, n: int):
    s = sessions.get(sid) or _404()
    if n >= len(s["events"]):
        _404()
    return Response(s["events"][n]["image"], media_type="image/jpeg")


@app.delete("/session/{sid}", status_code=204)
async def delete(sid: str):
    sessions.pop(sid, None)
    return Response(status_code=204)


def _404():
    raise HTTPException(404, "no such session")


app.mount("/static", StaticFiles(directory=STATIC), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

- [ ] **Step 2: Run it**

```bash
poetry run python scripts/mock_api.py
```

Expected: `Uvicorn running on http://127.0.0.1:8000`. `curl -s localhost:8000/health` → `{"ok":true,"sessions":0}`. `GET /` returns 404 until Task 2 creates `static/index.html` — that is fine.

- [ ] **Step 3: Commit**

```bash
git add scripts/mock_api.py
git commit -m "chore: mock api for client development"
```

---

### Task 2: Page structure

**Files:**
- Create: `static/index.html`

**Interfaces:**
- Produces: element ids that `app.js` binds to: `video`, `canvas`, `rule`, `start`, `stop`, `sample`, `sampleValue`, `reading`, `status`, `gate`, `state`, `evidence`, `events`, `toast`.

- [ ] **Step 1: Write the page**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Watcher</title>
<link rel="stylesheet" href="/static/style.css">
</head>
<body>
<main class="app">
  <header class="top">
    <h1>Watcher</h1>
    <p class="tagline">Point the camera. Say what should happen. Get told when it does.</p>
  </header>

  <section class="stage">
    <div class="viewfinder">
      <video id="video" autoplay playsinline muted></video>
      <canvas id="canvas" hidden></canvas>
      <div class="hud">
        <span id="gate" class="pill pill-idle">camera off</span>
        <span id="state" class="pill pill-idle">no rule</span>
      </div>
    </div>

    <form id="ruleForm" class="rule">
      <label for="rule">Tell me when…</label>
      <div class="row">
        <input id="rule" type="text" placeholder="the cat jumps onto the table" autocomplete="off" required>
        <button id="start" type="submit">Watch</button>
        <button id="stop" type="button" class="secondary" hidden>Stop</button>
      </div>
      <p id="reading" class="reading" hidden></p>
      <p id="status" class="status">Allow the camera to begin.</p>
      <label class="slider">every <output id="sampleValue">1.0</output>s
        <input id="sample" type="range" min="250" max="5000" step="250" value="1000">
      </label>
    </form>
  </section>

  <section class="log">
    <h2>What happened</h2>
    <p id="evidence" class="evidence">—</p>
    <ol id="events" class="events"></ol>
  </section>
</main>
<div id="toast" class="toast" hidden></div>
<script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Check it renders**

With the mock running, open `http://localhost:8000/`. Expected: unstyled page with the video box (black), the form and the log headings; no console errors except a 404 for `style.css` and `app.js` (they come next).

- [ ] **Step 3: Commit**

```bash
git add static/index.html
git commit -m "feat: client page structure"
```

---

### Task 3: Camera, sampling loop and API calls

**Files:**
- Create: `static/app.js`

**Interfaces:**
- Consumes: the HTTP contract; element ids from Task 2.
- Produces: the running page. State machine: `idle` → (camera granted) `ready` → (rule accepted) `watching` → `idle` on Stop or on a 404 from the server (session expired).

- [ ] **Step 1: Write the script**

```javascript
// static/app.js — camera → frames → server → status. No framework.
'use strict';

const $ = (id) => document.getElementById(id);
const video = $('video'), canvas = $('canvas'), form = $('ruleForm'), rule = $('rule');
const startBtn = $('start'), stopBtn = $('stop'), sample = $('sample'), sampleValue = $('sampleValue');
const reading = $('reading'), status = $('status'), gatePill = $('gate'), statePill = $('state');
const evidence = $('evidence'), events = $('events'), toast = $('toast');

const FRAME_WIDTH = 640, JPEG_QUALITY = 0.8;
let session = null;      // {session_id, predicate, direction}
let timer = null;        // setTimeout handle for the sampling loop
let inFlight = false;    // one upload at a time; a slow network must not pile up requests
let knownEvents = 0;

// ---------- camera ----------
async function openCamera() {
  const constraints = { video: { facingMode: 'environment', width: { ideal: 1280 } }, audio: false };
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (e) {
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  }
  await new Promise((r) => (video.onloadedmetadata = r));
  canvas.width = FRAME_WIDTH;
  canvas.height = Math.round((video.videoHeight / video.videoWidth) * FRAME_WIDTH);
  setPill(gatePill, 'ready', 'idle');
  say('Camera on. Describe what should happen, then press Watch.');
}

function grabJpeg() {
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((r) => canvas.toBlob(r, 'image/jpeg', JPEG_QUALITY));
}

// ---------- api ----------
async function api(path, init) {
  const res = await fetch(path, init);
  if (res.status === 204) return null;
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw Object.assign(new Error(body.hint || body.error || res.statusText), { status: res.status, body });
  return body;
}

// ---------- loop ----------
async function tick() {
  if (!session) return;
  if (!inFlight) {
    inFlight = true;
    try {
      const fd = new FormData();
      fd.append('frame', await grabJpeg(), 'frame.jpg');
      render(await api(`/session/${session.session_id}/frame`, { method: 'POST', body: fd }));
    } catch (e) {
      if (e.status === 404) { stop('Session expired — start again.'); return; }
      say(`Upload failed: ${e.message}`, true);
    } finally {
      inFlight = false;
    }
  }
  timer = setTimeout(tick, Number(sample.value));
}

async function start(ruleText) {
  startBtn.disabled = true;
  say('Understanding the rule…');
  try {
    session = await api('/session', {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ rule: ruleText }),
    });
  } catch (e) {
    startBtn.disabled = false;
    if (e.status === 503) { say('The room is full — 10 people are already watching. Try again in a minute.', true); return; }
    say(e.message, true);
    return;
  }
  startBtn.disabled = false;
  startBtn.hidden = true; stopBtn.hidden = false; rule.disabled = true;
  const arrow = session.direction === 'rising' ? 'becomes true' : 'becomes false';
  reading.textContent = `Watching for: “${session.predicate}” → ${arrow}`;
  reading.hidden = false;
  events.innerHTML = ''; knownEvents = 0; evidence.textContent = '—';
  setPill(statePill, 'unknown', 'idle');
  say('Watching.');
  tick();
}

function stop(message) {
  clearTimeout(timer); timer = null;
  if (session) api(`/session/${session.session_id}`, { method: 'DELETE' }).catch(() => {});
  session = null;
  startBtn.hidden = false; stopBtn.hidden = true; rule.disabled = false;
  reading.hidden = true;
  setPill(gatePill, 'ready', 'idle'); setPill(statePill, 'no rule', 'idle');
  say(message || 'Stopped.');
}

// ---------- render ----------
const GATE_LABEL = { first: 'first frame', skip: 'quiet', change: 'change', light: 'light changed' };
const GATE_TONE = { first: 'send', skip: 'idle', change: 'warn', light: 'info' };

function render(s) {
  const label = s.sent ? 'asking the model' : s.busy ? 'model busy' : GATE_LABEL[s.gate] + (s.streak ? ` ×${s.streak}` : '');
  setPill(gatePill, label, s.sent ? 'send' : GATE_TONE[s.gate]);
  if (s.state === null) setPill(statePill, 'unknown', 'idle');
  else setPill(statePill, s.state ? 'TRUE' : 'false', s.state ? 'true' : 'false');
  if (s.evidence) evidence.textContent = `“${s.evidence}”`;
  if (s.events > knownEvents) { knownEvents = s.events; refreshEvents(); }
  if (s.fired) { flash(); showToast('Event! ' + session.predicate); }
}

async function refreshEvents() {
  const view = await api(`/session/${session.session_id}`);
  events.innerHTML = '';
  for (const e of [...view.events].reverse()) {
    const li = document.createElement('li');
    const img = document.createElement('img');
    img.src = `/session/${session.session_id}/events/${e.n}.jpg`;
    img.alt = e.text;
    const cap = document.createElement('div');
    cap.innerHTML = `<b>${escapeHtml(e.text)}</b><time>${new Date(e.at).toLocaleTimeString()}</time>`;
    li.append(img, cap);
    events.append(li);
  }
}

function setPill(el, text, tone) { el.textContent = text; el.className = `pill pill-${tone}`; }
function say(text, isError) { status.textContent = text; status.classList.toggle('error', !!isError); }
function flash() { document.body.classList.add('flash'); setTimeout(() => document.body.classList.remove('flash'), 600); }
let toastTimer;
function showToast(text) { toast.textContent = text; toast.hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => (toast.hidden = true), 4000); }
function escapeHtml(s) { return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

// ---------- wiring ----------
form.addEventListener('submit', (e) => { e.preventDefault(); if (!session) start(rule.value.trim()); });
stopBtn.addEventListener('click', () => stop());
sample.addEventListener('input', () => (sampleValue.textContent = (sample.value / 1000).toFixed(2).replace(/0$/, '')));
window.addEventListener('pagehide', () => { if (session) navigator.sendBeacon && fetch(`/session/${session.session_id}`, { method: 'DELETE', keepalive: true }); });

openCamera().catch((e) => say(`Camera unavailable: ${e.message}. Use https:// or localhost.`, true));
```

- [ ] **Step 2: Walk the mock script**

With the mock running, open `http://localhost:8000/`, allow the camera, type `the cat jumps onto the table`, press Watch. Expected over ~10 seconds, in the HUD: `asking the model` → `quiet` → `quiet` → `change ×1` → `asking the model` (state pill `false`, evidence “cat on the chair”) → `quiet` → `light changed ×1` → `change ×1` → `asking the model` with the page flashing, a toast, and one event in the log with a thumbnail of your own camera → `quiet`. Then it loops.

Also check: typing `a cat` → status shows the hint, no session. Typing `full` → the room-is-full message. Press Stop → pills reset, input enabled. Drag the slider → the `every …s` output updates and the next tick uses it.

- [ ] **Step 3: Commit**

```bash
git add static/app.js
git commit -m "feat: camera sampling loop and status rendering"
```

---

### Task 4: Styling

**Files:**
- Create: `static/style.css`

The requirement is "decent UX, not a plain page". One dark theme, one accent, the viewfinder as the hero, pills that change colour with state, a visible flash on an event. Everything below is the complete file.

- [ ] **Step 1: Write the stylesheet**

```css
/* static/style.css */
:root {
  --bg: #0e1116; --panel: #161b23; --line: #232a35; --text: #e8ecf1; --muted: #8b95a5;
  --accent: #ffb347; --true: #3ddc84; --false: #6b7a90; --warn: #ffb347; --info: #5aa9ff; --send: #c084fc; --error: #ff6b6b;
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--text); font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
body.flash { animation: flash .6s ease-out; }
@keyframes flash { from { background: #3a2f10; } to { background: var(--bg); } }

.app { max-width: 720px; margin: 0 auto; padding: 20px 16px 48px; }
.top h1 { margin: 0; font-size: 28px; letter-spacing: .5px; }
.top h1::before { content: "◉ "; color: var(--accent); }
.tagline { margin: 4px 0 18px; color: var(--muted); }

.stage { background: var(--panel); border: 1px solid var(--line); border-radius: 16px; overflow: hidden; }
.viewfinder { position: relative; background: #000; aspect-ratio: 16 / 9; }
.viewfinder video { width: 100%; height: 100%; object-fit: cover; display: block; }
.hud { position: absolute; left: 12px; right: 12px; bottom: 12px; display: flex; gap: 8px; flex-wrap: wrap; }

.pill { padding: 4px 10px; border-radius: 999px; font-size: 13px; font-weight: 600; letter-spacing: .3px;
        background: rgba(0,0,0,.55); backdrop-filter: blur(6px); border: 1px solid rgba(255,255,255,.12); color: var(--text); }
.pill-idle { color: var(--muted); }
.pill-warn { border-color: var(--warn); color: var(--warn); }
.pill-info { border-color: var(--info); color: var(--info); }
.pill-send { border-color: var(--send); color: var(--send); animation: pulse 1s infinite; }
.pill-true { background: var(--true); color: #052; border-color: var(--true); }
.pill-false { color: var(--false); }
@keyframes pulse { 50% { opacity: .55; } }

.rule { padding: 16px; }
.rule label { display: block; color: var(--muted); font-size: 13px; margin-bottom: 6px; }
.row { display: flex; gap: 8px; }
input[type="text"] { flex: 1; min-width: 0; padding: 12px 14px; border-radius: 10px; border: 1px solid var(--line);
                     background: var(--bg); color: var(--text); font-size: 16px; }
input[type="text"]:focus { outline: none; border-color: var(--accent); }
input[type="text"]:disabled { color: var(--muted); }
button { padding: 12px 18px; border-radius: 10px; border: 0; background: var(--accent); color: #241a05; font-weight: 700; font-size: 15px; cursor: pointer; }
button.secondary { background: var(--line); color: var(--text); }
button:disabled { opacity: .6; cursor: wait; }
.reading { margin: 10px 0 0; color: var(--accent); font-size: 14px; }
.status { margin: 10px 0 0; color: var(--muted); min-height: 1.4em; }
.status.error { color: var(--error); }
.slider { display: flex; align-items: center; gap: 10px; margin-top: 12px; color: var(--muted); font-size: 13px; }
.slider input { flex: 1; accent-color: var(--accent); }

.log { margin-top: 20px; }
.log h2 { font-size: 15px; color: var(--muted); font-weight: 600; margin: 0 0 6px; text-transform: uppercase; letter-spacing: 1px; }
.evidence { margin: 0 0 12px; font-style: italic; color: var(--text); }
.events { list-style: none; margin: 0; padding: 0; display: grid; gap: 10px; }
.events li { display: grid; grid-template-columns: 120px 1fr; gap: 12px; align-items: center;
             background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 8px; }
.events img { width: 120px; aspect-ratio: 16 / 9; object-fit: cover; border-radius: 8px; display: block; }
.events time { display: block; color: var(--muted); font-size: 13px; margin-top: 2px; }

.toast { position: fixed; left: 50%; top: 18px; transform: translateX(-50%); background: var(--true); color: #052;
         padding: 10px 16px; border-radius: 999px; font-weight: 700; box-shadow: 0 6px 24px rgba(0,0,0,.4); z-index: 10; }

@media (max-width: 480px) {
  .row { flex-wrap: wrap; }
  button { flex: 1; }
  .events li { grid-template-columns: 96px 1fr; }
}
```

- [ ] **Step 2: Check on two widths**

Reload against the mock. Expected: dark page, the viewfinder fills the panel, pills float over the video bottom-left, the `TRUE` pill is solid green when it flips, an event fires with a yellow flash and a green toast, thumbnails render in the log. Narrow the window to 400 px: nothing overflows horizontally; buttons wrap under the input.

- [ ] **Step 3: Commit**

```bash
git add static/style.css
git commit -m "feat: client styling"
```

---

### Task 5: Integration with the real server

Prerequisite: the server plan's Task 7 is merged (`src/app.py` serves `static/`).

- [ ] **Step 1: Run the real server**

```bash
make dev
```

Open `http://localhost:8000/`. Type a real rule, e.g. `a hand appears in front of the camera`. Expected: the reading line shows Grok's predicate and direction; the HUD goes `asking the model` on the first frame, then `quiet` while nothing moves. Put your hand in view and hold it: `change ×1` → `asking the model` → state `TRUE` → flash + toast + event with the frame. Remove the hand: state goes `false` after two confirming calls; no event (rising rule). Hand back: a second event.

- [ ] **Step 2: Check a failure path**

Stop the server while watching. Expected: status shows `Upload failed: …` in red, the loop keeps trying, and when the server returns the page recovers or reports `Session expired — start again.` (the session was swept after 30 s).

- [ ] **Step 3: Commit nothing unless something needed fixing**

If integration required a client change, commit it as `fix: client — <what>`.

---

## Self-review

- **Spec coverage:** camera via `getUserMedia` with environment-facing preference; 640 px JPEG q0.8 on `SAMPLE_MS` with a 250–5000 slider; rule submit with `not_a_transition` and `full` handling; echo of predicate + direction; status strip rendering every `FrameStatus` field (`gate`, `streak`, `sent`, `busy`, `state`, `evidence`, `fired`, `events`); event log via `SessionView` + `events/{n}.jpg`; delete on stop and on page hide; works at 400 px; no external assets.
- **Types:** element ids in Task 2 and Task 3 match one-to-one. API paths and JSON keys match the spec table and the mock.
- **Placeholders:** none — every file is complete.
