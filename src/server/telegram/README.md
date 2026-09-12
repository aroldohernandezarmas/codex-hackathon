# Telegram notifier

Delivers camera events to a user's phone as a photo with a caption, and lets the user
check status or stop notifications from the chat. Two files, no framework: the Bot API
is called directly over `httpx`.

| File | What it holds |
|---|---|
| `bot.py` | `Bot` — thin Bot API client: `get_me`, `get_updates`, `send_message`, `send_photo`, `answer_callback`, `set_commands`, `deep_link`, `qr_svg` |
| `notifier.py` | `TelegramNotifier` (sends the event), `handle()` (turns one update into a reply), `poll()` (background long-polling loop), message texts and buttons |

## How binding works

Nothing is stored between restarts. A chat is bound to a session, and dies with it.

1. `POST /session` returns `telegram_link` — `https://t.me/<bot>?start=<session_id>`.
2. The page shows it as a QR code (`GET /session/{id}/qr.svg`) or a link.
3. The phone opens the bot; Telegram sends `/start <session_id>` to the bot.
4. `poll()` receives it, `handle()` sets `session.chat_id`, the bot replies "Linked" with
   two inline buttons: **Status** and **Stop**.
5. When the engine fires an event, `TelegramNotifier.notify()` looks the session up by
   id and sends the proof frame to that chat. No chat bound → log line only.

One chat per session. A second scan of the same QR replaces the first chat.

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
notifier = TelegramNotifier(bot, store)          # store: SessionStore

# at startup, inside the running event loop
await bot.get_me()
task = asyncio.create_task(poll(bot, store))     # runs until cancelled

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

| Command / button | Reply |
|---|---|
| `/start` | welcome with the three steps |
| `/start <id>` | binds the chat, "Linked" + buttons |
| `/status`, **Status** | rule, whether it is true now, what the model saw, event count |
| `/stop`, **Stop** | unbinds the chat |

Unknown session → "That session is gone, reload and scan again". Not bound → "Open the
page and scan its QR code". Messages are HTML; user text goes through `html.escape`.

## Event message

Photo = the frame that fired the event. Caption:

```
🔔 <rule>
<what the model saw>
#<n> · HH:MM:SS UTC
[👁 Status] [🔕 Stop]
```

## Limits and upgrade paths

- **Long-polling, one instance.** Telegram returns `409 Conflict` if two processes poll
  the same token (for example local dev and Render at once). Use a second bot for local
  work. If the service ever runs more than one process, replace `poll()` with a webhook:
  `setWebhook` on startup and a `POST /telegram` route that feeds updates to `handle()`.
- **Binding lives in memory.** Server restart or session TTL (30 s without frames) drops
  it; the user scans again.
- **One chat per session.** Several recipients → `chat_ids: list[int]` on `Session` and a
  loop in `notify()`.
- **Bot API errors** during send or reply are logged as warnings and never reach the
  frame request; a broken Telegram never slows the camera loop.

## Tests

`tests/test_telegram.py` — deep link and QR, bind / status / stop through `handle()`,
photo sent only when bound. Bot API is mocked with `httpx.MockTransport`; no token needed.

```bash
poetry run pytest tests/test_telegram.py -v
```
