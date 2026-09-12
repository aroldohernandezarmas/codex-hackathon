// static/app.js — camera → frames → server → status. No framework.
'use strict';

const $ = (id) => document.getElementById(id);
const video = $('video'), canvas = $('canvas'), form = $('ruleForm'), rule = $('rule');
const startBtn = $('start'), stopBtn = $('stop'), sample = $('sample'), sampleValue = $('sampleValue');
const reading = $('reading'), status = $('status'), gatePill = $('gate'), statePill = $('state');
const evidence = $('evidence'), events = $('events'), toast = $('toast');
const usage = $('usage'), resetUsageBtn = $('resetUsage');

const FRAME_WIDTH = 640, JPEG_QUALITY = 0.8;
const MAX_OUTSTANDING = 2; // at most this many uploads in flight at once
let session = null;      // {session_id, predicate, direction}
let timer = null;        // setTimeout handle for the sampling loop
let outstanding = 0;     // uploads currently in flight
let uploadSeq = 0;       // increasing tag for each upload, to detect out-of-order responses
let lastRenderedSeq = -1;
let knownEvents = 0;

// ---------- camera ----------
function releaseCamera() {
  if (video.srcObject) {
    video.srcObject.getTracks().forEach((t) => t.stop());
    video.srcObject = null;
  }
}

async function openCamera() {
  const constraints = { video: { facingMode: 'environment', width: { ideal: 1280 } }, audio: false };
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
  // Scheduled first: cadence is wall-clock, independent of how long the upload takes.
  timer = setTimeout(tick, Number(sample.value));
  if (!session) return;
  if (outstanding >= MAX_OUTSTANDING) return; // already at the concurrency cap — skip this sample
  outstanding++;
  const mySeq = ++uploadSeq;
  try {
    const fd = new FormData();
    fd.append('frame', await grabJpeg(), 'frame.jpg');
    const s = await api(`/session/${session.session_id}/frame`, { method: 'POST', body: fd });
    // Concurrent uploads can resolve out of order — drop a response older than
    // the newest one already rendered.
    if (mySeq > lastRenderedSeq) {
      lastRenderedSeq = mySeq;
      render(s);
    }
  } catch (e) {
    if (e.status === 404) { stop('Session expired — start again.'); return; }
    say(`Upload failed: ${e.message}`, true);
  } finally {
    outstanding--;
  }
}

async function start(ruleText) {
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
  say('Understanding the rule…');
  try {
    session = await api('/session', {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ rule: ruleText }),
    });
  } catch (e) {
    startBtn.disabled = false;
    if (e.status === 503) { say('The room is full right now. Try again in a minute.', true); return; }
    say(e.message, true);
    return;
  }
  startBtn.disabled = false;
  startBtn.hidden = true; stopBtn.hidden = false; rule.disabled = true;
  resetUsageBtn.hidden = false; renderUsage(session.usage);
  const arrow = session.direction === 'rising' ? 'becomes true' : 'becomes false';
  reading.textContent = `Watching for: “${session.predicate}” → ${arrow}`;
  reading.hidden = false;
  events.innerHTML = ''; knownEvents = 0; evidence.textContent = '—';
  outstanding = 0; uploadSeq = 0; lastRenderedSeq = -1;
  setPill(statePill, 'unknown', 'idle');
  say('Watching.');
  tick();
}

function stop(message) {
  clearTimeout(timer); timer = null;
  if (session) api(`/session/${session.session_id}`, { method: 'DELETE' }).catch(() => {});
  session = null;
  releaseCamera();
  startBtn.hidden = false; stopBtn.hidden = true; rule.disabled = false;
  reading.hidden = true; resetUsageBtn.hidden = true;
  setPill(gatePill, 'camera off', 'idle'); setPill(statePill, 'no rule', 'idle');
  say(message || 'Stopped.');
}

// ---------- render ----------
const GATE_LABEL = { first: 'first frame', skip: 'quiet', change: 'change', light: 'light changed' };
const GATE_TONE = { first: 'send', skip: 'idle', change: 'warn', light: 'info' };

function render(s) {
  if (!session) return; // response arrived after Stop cleared the session — nothing to render
  const label = s.sent ? 'asking the model' : s.busy ? 'model busy' : GATE_LABEL[s.gate] + (s.streak ? ` ×${s.streak}` : '');
  setPill(gatePill, label, s.sent ? 'send' : GATE_TONE[s.gate]);
  if (s.state === null) setPill(statePill, 'unknown', 'idle');
  else setPill(statePill, s.state ? 'TRUE' : 'false', s.state ? 'true' : 'false');
  if (s.evidence) evidence.textContent = `“${s.evidence}”`;
  if (s.events > knownEvents) refreshEvents(s.events);
  if (s.fired) { flash(); showToast('Event! ' + session.predicate); }
  if (s.usage) renderUsage(s.usage);
}

function renderUsage(u) {
  usage.textContent = `${u.total} (${u.prompt} in / ${u.completion} out, ${u.calls} calls)`;
  usage.title = 'API tokens spent by this session';
}

async function resetUsage() {
  if (!session) return;
  try { renderUsage(await api(`/session/${session.session_id}/usage/reset`, { method: 'POST' })); }
  catch (e) { say(`Reset failed: ${e.message}`, true); }
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
resetUsageBtn.addEventListener('click', resetUsage);
sample.addEventListener('input', () => (sampleValue.textContent = (sample.value / 1000).toFixed(2).replace(/0$/, '')));
window.addEventListener('pagehide', () => { if (session) navigator.sendBeacon && fetch(`/session/${session.session_id}`, { method: 'DELETE', keepalive: true }); releaseCamera(); });

openCamera().catch((e) => say(`Camera unavailable: ${e.message}. Use https:// or localhost.`, true));
