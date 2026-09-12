const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { test } = require('node:test');

function page() {
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, {
      textContent: '', value: '1000', disabled: false, dataset: {},
      classList: { add() {}, remove() {}, toggle() {} },
      addEventListener() {}, removeAttribute(name) { delete this[name]; },
    });
    return elements.get(id);
  };
  const context = vm.createContext({
    document: { getElementById: element, body: { classList: { add() {}, remove() {} } } },
    navigator: { mediaDevices: { getUserMedia: () => new Promise(() => {}) } },
    window: { addEventListener() {} },
    localStorage: { getItem() { return null; }, setItem() {} },
    fetch: () => new Promise(() => {}),
    setTimeout() { return 1; }, clearTimeout() {},
    setInterval() { return 1; }, clearInterval() {},
    EventSource: class { constructor(url) { this.url = url; } addEventListener() {} close() { this.closed = true; } },
    FormData: class { append() {} }, Date,
  });
  const run = (code) => vm.runInContext(code, context);
  run(fs.readFileSync('static/app.js', 'utf8'));
  return { context, element, run };
}

for (const outcome of ['success', '404']) {
  test(`old upload ${outcome} cannot affect a replacement session`, async () => {
    const { context, element, run } = page();
    run(`
      grabJpeg = async () => ({}); video.readyState = 2;
      session = { session_id: 'old' };
      api = () => new Promise((resolve, reject) => { globalThis.resolveUpload = resolve; globalThis.rejectUpload = reject; });
      globalThis.pending = tick();
    `);
    await Promise.resolve();
    run("session = { session_id: 'new' }; outstanding = 0; uploadSeq = 0; lastRenderedSeq = -1;");
    if (outcome === '404') context.rejectUpload({ status: 404 });
    else context.resolveUpload({ state: true, evidence: 'old result', events: 0, gate: 'skip' });
    await context.pending;
    assert.equal(run('session.session_id'), 'new');
    assert.equal(run('outstanding'), 0);
    assert.equal(element('evidence').textContent, '');
  });
}

test('stop during JPEG capture does not upload to a replacement session', async () => {
  const { context, run } = page();
  run(`
    session = { session_id: 'old' }; video.readyState = 2;
    grabJpeg = () => new Promise(r => { globalThis.finishCapture = r; });
    globalThis.uploads = 0; api = async () => { globalThis.uploads++; };
    globalThis.pending = tick();
    session = { session_id: 'new' }; outstanding = 0;
  `);
  context.finishCapture({});
  await context.pending;
  assert.equal(context.uploads, 0);
  assert.equal(run('outstanding'), 0);
});

test('Telegram subscription survives Stop and is attached to the next session', async () => {
  const { element, run } = page();
  run(`
    showSubscription({ token: 'browser', telegram_link: 'https://t.me/cam?start=browser', linked: true });
    video.srcObject = { getTracks: () => [] };
    globalThis.createdBodies = [];
    api = async (path, init) => {
      if (init.method === 'DELETE') return;
      createdBodies.push(JSON.parse(init.body));
      return { session_id: 'new', predicate: 'present', direction: 'rising', usage: {total: 0, usd: 0} };
    };
  `);
  await run("start('arrives')");
  assert.equal(run('createdBodies[0].subscriber'), 'browser');
  await run('stop()');
  assert.equal(element('notify').hidden, false);
  assert.equal(element('qrLink').href, 'https://t.me/cam?start=browser');
  assert.equal(element('qrImg').src, '/subscriber/browser/qr.svg');
  run('video.srcObject = { getTracks: () => [] };');
  await run("start('leaves')");
  assert.equal(run('createdBodies[1].subscriber'), 'browser');
});

test('Stop cancels pending session creation and deletes its eventual result', async () => {
  const { context, run } = page();
  run(`
    video.srcObject = { getTracks: () => [] };
    globalThis.deleted = [];
    api = (path, init) => init.method === 'DELETE'
      ? (deleted.push(path), Promise.resolve())
      : new Promise(r => { globalThis.finishCreate = r; });
    globalThis.pending = start('arrives');
    stop();
  `);
  context.finishCreate({ session_id: 'late' });
  await context.pending;
  assert.equal(run('session'), null);
  assert.deepEqual(Array.from(context.deleted), ['/session/late']);
});

test('Reset waits for deletion and leaves usable controls if creation fails', async () => {
  const { context, element, run } = page();
  run(`
    session = { session_id: 'old' };
    video.srcObject = { getTracks: () => [] };
    globalThis.creations = 0;
    api = (path, init) => init.method === 'DELETE'
      ? new Promise(r => { globalThis.finishDelete = r; })
      : (globalThis.creations++, Promise.reject({status: 503}));
    globalThis.pending = restart();
  `);
  assert.equal(context.creations, 0);
  context.finishDelete();
  await context.pending;
  assert.equal(context.creations, 1);
  assert.equal(run('session'), null);
  assert.equal(element('start').hidden, false);
  assert.equal(element('start').disabled, false);
  assert.equal(element('rule').disabled, false);
});

test('push delivers event immediately, rejects stale frame status and closes on Stop', () => {
  const { element, run } = page();
  run(`
    session = { session_id: 'new', predicate: 'present' };
    globalThis.refreshes = 0; refreshEvents = () => { refreshes++; };
    connectUpdates(); globalThis.source = updates;
    source.onmessage({data: JSON.stringify({revision: 2, state: true, evidence: 'fresh', events: 1})});
  `);
  assert.equal(element('state').textContent, 'TRUE');
  assert.equal(element('toast').textContent, 'Event! present');
  run("renderDetection({revision: 1, state: false, evidence: 'stale', events: 0});");
  assert.equal(element('evidence').textContent, '“fresh”');
  run("api = async () => {}; stop(); source.onmessage({data: JSON.stringify({revision: 3, state: false, events: 0})});");
  assert.equal(run('source.closed'), true);
  assert.equal(element('state').textContent, 'no rule');
});
