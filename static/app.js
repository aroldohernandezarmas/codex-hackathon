// static/app.js — camera → frames → server → status. No framework.
'use strict';

const $ = (id) => document.getElementById(id);
const video = $('video'), canvas = $('canvas'), form = $('ruleForm'), rule = $('rule');
const startBtn = $('start'), stopBtn = $('stop'), restartBtn = $('restart'), sample = $('sample'), sampleValue = $('sampleValue');
const reading = $('reading'), status = $('status'), gatePill = $('gate'), statePill = $('state'), flip = $('flip');
const evidence = $('evidence'), events = $('events'), toast = $('toast');
const notify = $('notify'), qrLink = $('qrLink'), qrImg = $('qrImg');
const qrFallback = $('qrFallback'), qrBadge = $('qrBadge'), qrHint = $('qrHint');

const SUB_KEY = 'watcher.subscriber'; // the token survives reloads, so one scan is enough
const SUB_POLL_MS = 5000;
const usage = $('usage'), cost = $('cost'), elapsed = $('elapsed');
let generation = 0;
let updates = null, lastRevision = -1, announcedEvents = 0;
let startedAt = 0, clock = null;
function showElapsed() {
  const s = Math.floor((Date.now() - startedAt) / 1000);
  elapsed.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

const FRAME_WIDTH = 640, JPEG_QUALITY = 0.8;
const MAX_OUTSTANDING = 2; // at most this many uploads in flight at once
let session = null;      // {session_id, predicate, direction}
let timer = null;        // setTimeout handle for the sampling loop
let outstanding = 0;     // uploads currently in flight
let uploadSeq = 0;       // increasing tag for each upload, to detect out-of-order responses
let lastRenderedSeq = -1;
let knownEvents = 0;
let facing = 'environment'; // which camera to ask for; survives stop/start

// ---------- camera ----------
function releaseCamera() {
  if (video.srcObject) {
    video.srcObject.getTracks().forEach((t) => t.stop());
    video.srcObject = null;
  }
}

async function openCamera() {
  const constraints = { video: { facingMode: { ideal: facing }, width: { ideal: 1280 } }, audio: false };
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (firstError) {
    try {
      video.srcObject = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    } catch (e) {
      throw firstError; // real cause (e.g. permission denial) beats the fallback's error
    }
  }
  if (video.readyState < 1) { // HAVE_NOTHING — metadata hasn't loaded yet
    await new Promise((r) => (video.onloadedmetadata = r));
  }
  // Selfie view is mirrored for the preview only — the uploaded JPEG keeps the true orientation.
  video.classList.toggle('mirrored', facing === 'user');
  canvas.width = FRAME_WIDTH;
  canvas.height = Math.round((video.videoHeight / video.videoWidth) * FRAME_WIDTH);
  // While a session runs, render() owns the gate pill and the status line — a mid-watch
  // flip reopens the camera and must not overwrite them.
  if (!session) {
    setPill(gatePill, 'ready', 'idle');
    say('Camera on. Describe what should happen, then press Watch.');
  }
}

// The flip button only makes sense with more than one camera. Device kinds are readable
// without permission; we still call this after the first open so nothing is enumerated early.
async function revealFlipIfMultiCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  const devices = await navigator.mediaDevices.enumerateDevices().catch(() => []);
  flip.hidden = devices.filter((d) => d.kind === 'videoinput').length < 2;
}

async function flipCamera() {
  if (flip.disabled) return; // a swap is already in flight
  flip.disabled = true;
  const previous = facing;
  facing = facing === 'environment' ? 'user' : 'environment';
  releaseCamera();
  try {
    await openCamera();
  } catch (e) {
    facing = previous; // the requested camera isn't available — go back to the one that worked
    try {
      await openCamera();
    } catch (_) {
      // both gone; the message below is the only thing left to say
    }
    say(`Could not switch camera: ${e.message}`, true);
  } finally {
    flip.disabled = false;
  }
}

function grabJpeg() {
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((r) => canvas.toBlob(r, 'image/jpeg', JPEG_QUALITY));
}

// ---------- telegram subscription ----------
// The token is the binding code, and it outlives every watch - so the QR can be shown
// before any rule exists, and one scan covers every watch this browser starts.
let subscriber = null;

async function subscribe() {
  const saved = localStorage.getItem(SUB_KEY);
  if (saved) {
    try {
      return await api(`/subscriber/${saved}`); // still known to this server?
    } catch (e) {
      if (e.status !== 404) throw e; // 404: expired, or a server restart wiped it
    }
  }
  const fresh = await api('/subscriber', { method: 'POST' });
  localStorage.setItem(SUB_KEY, fresh.token);
  return fresh;
}

// The QR stays on screen whether or not the chat is linked - only the badge changes.
// It is hidden in exactly one case: no bot is configured, so there is nothing to offer.
function showSubscription(s) {
  subscriber = s.token;
  if (!s.telegram_link) {
    notify.hidden = true;
    return;
  }
  qrLink.href = s.telegram_link;
  const src = `/subscriber/${s.token}/qr.svg`;
  // A failed image is retried on the next poll; comparing against the token rather than
  // the full src keeps a cache-busted retry from looping.
  if (qrImg.dataset.token !== s.token || qrImg.dataset.failed === '1') {
    qrImg.dataset.token = s.token;
    delete qrImg.dataset.failed;
    qrImg.src = qrImg.dataset.retry ? `${src}?r=${Date.now()}` : src;
  }
  setPill(qrBadge, s.linked ? '✓ linked' : 'not linked', s.linked ? 'true' : 'idle');
  qrHint.textContent = s.linked
    ? 'Events go to your Telegram chat.'
    : 'Scan to get events in Telegram.';
  notify.hidden = false;
}

// An unreachable QR must not leave a broken image on a white plate - fall back to a
// plain link, which is what the phone running this page would tap anyway.
qrImg.addEventListener('error', () => {
  if (!qrImg.getAttribute('src')) return; // src cleared, not a real failure
  qrImg.dataset.failed = '1';
  qrImg.dataset.retry = '1';
  qrImg.hidden = true;
  qrFallback.hidden = false;
});
qrImg.addEventListener('load', () => {
  qrImg.hidden = false;
  qrFallback.hidden = true;
  delete qrImg.dataset.retry;
});

async function watchSubscription() {
  try {
    showSubscription(await subscribe());
  } catch (e) {
    notify.hidden = true; // no channel to offer - the page still works without one
  }
  setTimeout(watchSubscription, SUB_POLL_MS);
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
  // Scheduled first: cadence is wall-clock, independent of how long the upload takes.
  timer = setTimeout(tick, Number(sample.value));
  if (!session) return;
  if (outstanding >= MAX_OUTSTANDING) return; // already at the concurrency cap — skip this sample
  if (video.readyState < 2) return; // HAVE_CURRENT_DATA — mid camera swap, no frame to grab
  const currentSession = session;
  outstanding++;
  const mySeq = ++uploadSeq;
  try {
    const fd = new FormData();
    const jpeg = await grabJpeg();
    if (session !== currentSession) return;
    fd.append('frame', jpeg, 'frame.jpg');
    const s = await api(`/session/${currentSession.session_id}/frame`, { method: 'POST', body: fd });
    // Concurrent uploads can resolve out of order — drop a response older than
    // the newest one already rendered.
    if (session !== currentSession) return;
    if (mySeq > lastRenderedSeq) {
      lastRenderedSeq = mySeq;
      render(s);
    }
  } catch (e) {
    if (session !== currentSession) return;
    if (e.status === 404) { stop('Session expired — start again.'); return; }
    say(`Upload failed: ${e.message}`, true);
  } finally {
    if (session === currentSession) outstanding--;
  }
}

async function start(ruleText) {
  if (startBtn.disabled) return;
  const attempt = ++generation;
  startBtn.disabled = true;
  if (!video.srcObject) {
    try {
      await openCamera();
    } catch (e) {
      startBtn.disabled = false;
      say(`Camera unavailable: ${e.message}. Use https:// or localhost.`, true);
      return;
    }
  }
  if (attempt !== generation) return;
  say('Understanding the rule…');
  try {
    const created = await api('/session', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ rule: ruleText, subscriber: subscriber }),
    });
    if (attempt !== generation) {
      api(`/session/${created.session_id}`, { method: 'DELETE' }).catch(() => {});
      return;
    }
    session = created;
  } catch (e) {
    if (attempt !== generation) return;
    startBtn.disabled = false;
    if (e.status === 503) { say('The room is full right now. Try again in a minute.', true); return; }
    say(e.message, true);
    return;
  }
  startBtn.disabled = false;
  startBtn.hidden = true; stopBtn.hidden = false; restartBtn.hidden = false; rule.disabled = true;
  lastRevision = -1; announcedEvents = 0;
  renderUsage(session.usage);
  startedAt = Date.now(); showElapsed(); clock = setInterval(showElapsed, 1000);
  const arrow = session.direction === 'rising' ? 'becomes true' : 'becomes false';
  reading.textContent = `Watching for: “${session.predicate}” → ${arrow}`;
  reading.hidden = false;
  events.innerHTML = ''; knownEvents = 0; evidence.textContent = '—';
  outstanding = 0; uploadSeq = 0; lastRenderedSeq = -1;
  setPill(statePill, 'unknown', 'idle');
  say('Watching.');
  connectUpdates();
  tick();
}

async function restart() {
  if (!session) return;
  const deletion = stop(undefined, true);
  const attempt = generation;
  await deletion;
  if (attempt === generation) await start(rule.value.trim());
}

function stop(message, keepCamera = false) {
  generation++;
  if (updates) { updates.close(); updates = null; }
  clearTimeout(timer); timer = null;
  const deletion = session ? api(`/session/${session.session_id}`, { method: 'DELETE' }).catch(() => {}) : Promise.resolve();
  session = null;
  if (!keepCamera) releaseCamera();
  startBtn.disabled = false;
  startBtn.hidden = false; stopBtn.hidden = true; restartBtn.hidden = true; rule.disabled = false;
  reading.hidden = true;
  clearInterval(clock); clock = null;
  setPill(gatePill, keepCamera ? 'ready' : 'camera off', 'idle'); setPill(statePill, 'no rule', 'idle');
  say(message || 'Stopped.');
  return deletion;
}

// ---------- render ----------
const GATE_LABEL = { first: 'first frame', skip: 'quiet', change: 'change', light: 'light changed' };
const GATE_TONE = { first: 'send', skip: 'idle', change: 'warn', light: 'info' };

function render(s) {
  if (!session) return; // response arrived after Stop cleared the session — nothing to render
  const label = s.sent ? 'asking the model' : s.busy ? 'model busy' : GATE_LABEL[s.gate] + (s.streak ? ` ×${s.streak}` : '');
  setPill(gatePill, label, s.sent ? 'send' : GATE_TONE[s.gate]);
  renderDetection(s);
}

function connectUpdates() {
  const currentSession = session;
  updates = new EventSource(`/session/${session.session_id}/updates`);
  updates.onmessage = (event) => {
    if (session === currentSession) renderDetection(JSON.parse(event.data));
  };
  updates.addEventListener('expired', () => {
    if (session === currentSession) stop('Session expired — start again.');
  });
}

function renderDetection(s) {
  if (!session || (s.revision !== undefined && s.revision < lastRevision)) return;
  if (s.revision !== undefined) lastRevision = s.revision;
  if (s.state === null) setPill(statePill, 'unknown', 'idle');
  else setPill(statePill, s.state ? 'TRUE' : 'false', s.state ? 'true' : 'false');
  evidence.textContent = s.evidence ? `“${s.evidence}”` : '—';
  if (s.events > knownEvents) refreshEvents(s.events);
  if (s.events > announcedEvents) {
    announcedEvents = s.events;
    flash(); showToast('Event! ' + session.predicate);
  }
  if (s.usage) renderUsage(s.usage);
}

function renderUsage(u) {
  usage.textContent = u.total.toLocaleString();
  usage.title = `${u.prompt} in / ${u.completion} out, ${u.calls} calls`;
  cost.textContent = u.usd < 0.01 ? `$${u.usd.toFixed(4)}` : `$${u.usd.toFixed(2)}`;
}

// Fetches the event list and only advances knownEvents once it actually succeeds,
// so a Stop or a failed request mid-refresh never drops an event or throws unhandled.
async function refreshEvents(count) {
  if (!session) return;
  const sid = session.session_id;
  try {
    const view = await api(`/session/${sid}`);
    if (!session || session.session_id !== sid) return; // stopped (or restarted) mid-refresh
    events.innerHTML = '';
    for (const e of [...view.events].reverse()) {
      const li = document.createElement('li');
      const img = document.createElement('img');
      img.src = `/session/${sid}/events/${e.n}.jpg`;
      img.alt = e.text;
      const cap = document.createElement('div');
      const b = document.createElement('b');
      b.textContent = e.text;
      const time = document.createElement('time');
      time.textContent = new Date(e.at).toLocaleTimeString();
      cap.append(b, time);
      li.append(img, cap);
      events.append(li);
    }
    knownEvents = count;
  } catch (e) {
    // Leave knownEvents alone — the next status with more events retries the refresh.
  }
}

function setPill(el, text, tone) { el.textContent = text; el.className = `pill pill-${tone}`; }
function say(text, isError) { status.textContent = text; status.classList.toggle('error', !!isError); }
function flash() { document.body.classList.add('flash'); setTimeout(() => document.body.classList.remove('flash'), 600); }
let toastTimer;
function showToast(text) { toast.textContent = text; toast.hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => (toast.hidden = true), 4000); }

// ---------- wiring ----------
form.addEventListener('submit', (e) => { e.preventDefault(); if (!session) start(rule.value.trim()); });
stopBtn.addEventListener('click', () => stop());
restartBtn.addEventListener('click', restart);
flip.addEventListener('click', flipCamera);
function showSample() { sampleValue.textContent = String(Number(sample.value) / 1000); }
sample.addEventListener('input', showSample);
showSample();
window.addEventListener('pagehide', () => { if (updates) updates.close(); if (session) navigator.sendBeacon && fetch(`/session/${session.session_id}`, { method: 'DELETE', keepalive: true }); releaseCamera(); });

watchSubscription();
openCamera()
  .then(revealFlipIfMultiCamera)
  .catch((e) => say(`Camera unavailable: ${e.message}. Use https:// or localhost.`, true));
