"""Events to Telegram. Binding: the page shows a deep link (QR), the phone opens the bot,
the bot receives `/start <session_id>` and the chat is bound to that session.
Then the chat has two buttons and two commands: status and stop."""

import asyncio
from dataclasses import dataclass
from html import escape
from typing import Optional

import httpx
from loguru import logger

from src.server.notifier import Notifier
from src.server.session import Event, Session, SessionStore, Watch
from src.server.telegram.bot import Bot, Buttons

COMMANDS = {"status": "what the camera sees now", "stop": "stop notifications"}
BUTTONS: Buttons = [[("👁 Status", "status"), ("🔕 Stop", "stop")]]

WELCOME = (
    "👋 <b>Camera Events</b>\n"
    "I watch a camera and message you the moment something happens.\n\n"
    "1. Open the camera page\n"
    "2. Type what to watch for\n"
    "3. Scan the QR code shown there\n\n"
    "That's it — I'll send a photo when it happens."
)
GONE = "⌛ That session is gone. Reload the camera page and scan the QR code again."
NOT_LINKED = "🔗 Not linked to a camera yet. Open the page and scan its QR code."
STOPPED = "🔕 Stopped. Scan the QR code again to relink."


@dataclass
class Reply:
    chat_id: int
    text: str
    buttons: Optional[Buttons] = None
    callback_id: Optional[str] = None  # set when a button was pressed


def _state(w: Watch) -> str:
    if w.tracker.state is None:
        return "⏳ still looking"
    return "✅ yes" if w.tracker.state else "❌ no"


def linked_text(s: Session) -> str:
    return (
        f"✅ <b>Linked</b>\nWatching: <i>{escape(s.watch.rule)}</i>\n"
        "I'll send a photo the moment it happens."
    )


def status_text(s: Session) -> str:
    w = s.watch
    seen = escape(w.evidence) if w.evidence else "nothing yet"
    return (
        f"👁 <i>{escape(w.rule)}</i>\n"
        f"Is it true now: {_state(w)}\n"
        f"I see: {seen}\n"
        f"Events so far: {len(w.events)}"
    )


def caption(w: Watch, event: Event) -> str:
    return (
        f"🔔 <b>{escape(w.rule)}</b>\n"
        f"{escape(w.evidence) if w.evidence else escape(event.text)}\n"
        f"#{event.n + 1} · {event.at[11:19]} UTC"
    )


def by_chat(store: SessionStore, chat_id: int) -> Optional[Session]:
    # ponytail: linear scan, MAX_SESSIONS is 10
    return next((s for s in store.all() if s.chat_id == chat_id), None)


def handle(store: SessionStore, update: dict) -> Optional[Reply]:
    """One update -> what to answer, or None for anything we ignore."""
    callback = update.get("callback_query")
    if callback:
        chat_id = callback["message"]["chat"]["id"]
        return _command(store, chat_id, callback.get("data", ""), callback["id"])
    message = update.get("message") or {}
    text, chat = message.get("text") or "", message.get("chat") or {}
    if not text.startswith("/") or "id" not in chat:
        return None
    command, _, arg = text[1:].partition(" ")
    if command == "start":
        return _start(store, chat["id"], arg.strip())
    return _command(store, chat["id"], command)


def _start(store: SessionStore, chat_id: int, code: str) -> Reply:
    if not code:
        return Reply(chat_id, WELCOME)
    try:
        session = store.get(code)
    except KeyError:
        return Reply(chat_id, GONE)
    session.chat_id = chat_id
    return Reply(chat_id, linked_text(session), BUTTONS)


def _command(
    store: SessionStore, chat_id: int, command: str, callback_id: Optional[str] = None
) -> Optional[Reply]:
    if command not in COMMANDS:
        return None
    session = by_chat(store, chat_id)
    if session is None:
        return Reply(chat_id, NOT_LINKED, callback_id=callback_id)
    if command == "stop":
        session.chat_id = None
        return Reply(chat_id, STOPPED, callback_id=callback_id)
    return Reply(chat_id, status_text(session), BUTTONS, callback_id)


class TelegramNotifier(Notifier):
    def __init__(self, bot: Bot, store: SessionStore) -> None:
        self.bot = bot
        self.store = store

    async def notify(self, session_id: str, watch: Watch, event: Event) -> None:
        await super().notify(session_id, watch, event)  # keep the log line
        try:
            chat_id = self.store.get(session_id).chat_id
        except KeyError:
            return
        if chat_id is None:
            return
        try:
            await self.bot.send_photo(
                chat_id, event.image, caption(watch, event), BUTTONS
            )
        except httpx.HTTPError as e:
            logger.warning("session={} telegram send failed: {}", session_id, e)


async def poll(bot: Bot, store: SessionStore) -> None:
    """Background loop: long-poll updates, answer commands and buttons. Runs until cancelled."""
    # ponytail: long-polling, one instance only (Telegram rejects two pollers) - switch
    # to a webhook if the service ever scales past one process
    await bot.set_commands(COMMANDS)
    offset = 0
    while True:
        try:
            updates = await bot.get_updates(offset)
        except httpx.HTTPError as e:
            logger.warning("telegram poll failed: {}", e)
            await asyncio.sleep(3)
            continue
        for update in updates:
            offset = update["update_id"] + 1
            reply = handle(store, update)
            if reply is None:
                continue
            try:
                if reply.callback_id:
                    await bot.answer_callback(reply.callback_id)
                await bot.send_message(reply.chat_id, reply.text, reply.buttons)
            except httpx.HTTPError as e:
                logger.warning("telegram reply failed: {}", e)
