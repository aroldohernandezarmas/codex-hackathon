# Telegram notifier

Delivers camera events to a user's phone as a photo with a caption, and lets the user
take fresh snapshots, change rules, browse events and pause alerts from the chat. Two files, no framework: the Bot API
is called directly over `httpx`.

| File | What it holds |
|---|---|
| `bot.py` | `Bot` — thin Bot API client: `get_me`, `get_updates`, `send_message`, `send_photo`, `answer_callback`, `set_commands`, `deep_link`, `qr_svg`, `edit_message`, `chat_action` |
| `notifier.py` | `TelegramNotifier` (sends the event), `handle()` (menus and state), `respond()` (snapshots, rule normalization and delivery), `poll()` (background long-polling loop), message texts and buttons |

## How binding works

A chat is bound to a **subscriber** — a browser, not a watch. The binding therefore
outlives any single session, which is what lets the page show the QR before a rule
exists. Nothing is stored between restarts.

1. `POST /subscriber` mints a token (`Subscribers.create`, `src/server/session.py`) and
   returns `telegram_link` — `https://t.me/<bot>?start=<token>`. The page keeps the token
   in `localStorage`, so one scan covers every later watch from that browser.
2. The page shows the link as a QR code (`GET /subscriber/{token}/qr.svg`).
3. The phone opens the bot; Telegram sends `/start <token>` to the bot.
4. `poll()` receives it, `handle()` sets `subscriber.chat_id`, the bot opens the camera dashboard
   with **Snapshot now**, **Change rule**, **Recent events** and alert controls.
5. `POST /session` carries the token, so the new session records `session.subscriber`.
6. When the engine fires an event, `TelegramNotifier.notify()` resolves
   session → subscriber → chat and sends the proof frame there. No chat bound → log only.

The page polls `GET /subscriber/{token}` every 5 s; that `linked` flag is the only thing
driving whether the QR is on screen. It shows whenever the browser is not linked, and
disappears once it is.

One camera subscriber per private chat; linking another browser disconnects the previous binding. A second scan of the same QR replaces the first chat. A token
the server no longer knows (expired, or a restart) 404s, and the page mints a new one.

## Setup

1. Create a bot with [@BotFather](https://t.me/BotFather): `/newbot`, pick a name and a
   username ending in `bot`. Copy the token.
2. Put it in the environment:

   ```
   TELEGRAM_BOT_TOKEN=123456789:AAF...
   ```

   Locally that is `.env`; on Render it is the service's Environment tab (`render.yaml`
   declares the key with `sync: false`). Empty token = Telegram off, the base `Notifier`
   logs events and nothing else changes.
3. Start the server. On startup it calls `getMe` (to build deep links) and
   `setMyCommands` (the "/" menu), then starts polling.

Optional cosmetics in BotFather: `/setdescription`, `/setabouttext`, `/setuserpic`.

## Wiring into an app

`src/server/app.py` already does this; shown here for a different host:

```python
from src.server.telegram.bot import Bot
from src.server.telegram.notifier import TelegramNotifier, poll

bot = Bot(token)
subs = Subscribers(max_subscribers, ttl)         # browser -> chat, outlives sessions
notifier = TelegramNotifier(bot, store, subs)    # store: SessionStore

# at startup, inside the running event loop
await bot.get_me()
task = asyncio.create_task(poll(bot, store, subs, perception))   # runs until cancelled

# somewhere in the engine, when an event fires
await notifier.notify(session.id, session.watch, event)

# on shutdown
task.cancel()
await bot.aclose()
```

`TelegramNotifier` extends the base `Notifier` and calls `super().notify()` first, so the
log line stays. The engine depends only on `Notifier.notify(session_id, watch, event)`;
swap the implementation and nothing else moves.

## Chat commands

| Command / button | Behavior |
|---|---|
| `/start` | Welcome, or dashboard when connected |
| `/start <token>` | Connect this browser and open its dashboard |
| `/menu`, `/status`, **Dashboard** | Camera freshness, rule, observation, event count and alert state |
| `/snapshot`, **Snapshot now** | Wait up to 8 seconds for a newly received, valid frame and send it as a photo |
| `/rule`, **Change rule** | Ask for a new rule; `/rule The door opens` also works |
| `/events`, **Recent events** | Latest five events with buttons to open their photos |
| `/pause`, `/stop`, **Pause alerts** | Silence Telegram alerts while retaining the connection and recording events |
| `/resume`, **Resume alerts** | Send future event alerts again; no replay of paused alerts |
| `/help`, **How it works** | Explain controls and camera requirements |
| **Disconnect** | Ask for confirmation before removing the chat binding |

All interface copy is English. Text screens use HTML headings, block quotes and inline
buttons. Navigation edits an existing text message; photos are separate cards with
follow-up controls. Refreshing an unchanged screen is harmless. User/model text is
escaped and truncated to keep messages within Telegram limits. Camera controls are
restricted to private chats.

### Fresh snapshots

The browser must remain open and send frames for an active watch. Snapshot waits for
a new validated frame after the request, independently of the motion gate and model
calls. It never substitutes an old event photo. If no frame arrives within 8 seconds,
the bot shows camera recovery instructions and a retry button. The timestamp is the
server's UTC frame receipt time, not a hardware capture timestamp.

### Changing the watch

Normalization validates the replacement before applying it. Invalid rules and model
errors leave the old watch active. The session and its event history are preserved,
the tracker is reset, and the next available frame is analyzed using the new rule.
In-flight results for the old watch cannot append events after replacement. The
browser receives the new rule, predicate and direction through its existing SSE feed.
Cancel or navigate away to leave rule-entry mode.

## Event message

Photo = the frame that fired the event. Caption:

```
🔔 MOMENT DETECTED

▎ The cat jumps onto the table
A cat is standing on the table.

Event 01 · 2026-09-12 · 14:32:10 UTC
[📸 See now]      [🗂 Recent events]
[🔕 Pause alerts] [Dashboard]
```

History photos show their original event text, even after the rule changes. History
buttons are scoped to the connected session so an old card cannot open another watch.

## Limits and upgrade paths

- **Long-polling, one instance.** Telegram returns `409 Conflict` if two processes poll
  the same token (for example local dev and Render at once). Use a second bot for local
  work. If the service ever runs more than one process, replace `poll()` with a webhook:
  `setWebhook` on startup and a `POST /telegram` route that feeds updates to `respond()`.
  Updates currently run sequentially; snapshot waits and rule normalization delay later
  commands, while camera processing and event delivery remain independent.
- **Binding lives in memory.** A server restart drops it and the user scans again;
  `SUBSCRIBER_TTL` (default 24 h, refreshed by the page's poll) drops idle ones.
  Persisting it means a real store keyed by the token.
- **One chat per subscriber.** Several recipients → `chat_ids: list[int]` on `Subscriber`
  and a loop in `notify()`.
- **Bot API errors** during send or reply are logged as warnings and never reach the
  frame request; a broken Telegram never slows the camera loop.

## Tests

`tests/test_telegram.py` — deep link and QR, bind-before-any-rule / status / stop through
`handle()`, fresh-frame waits and timeout behavior, rule replacement and cancellation,
message editing, history isolation, private-chat controls and pause/resume. `tests/test_app.py` covers the
`/subscriber` routes. Bot API is mocked with `httpx.MockTransport`; no token needed.

```bash
poetry run pytest tests/test_telegram.py -v
```
