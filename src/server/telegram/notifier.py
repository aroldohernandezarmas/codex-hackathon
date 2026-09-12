"""Events to Telegram. Binding: the page shows a deep link (QR), the phone opens the bot,
the bot receives `/start <session_id>` and the chat is bound to that session."""

import asyncio
from typing import Optional

import httpx
from loguru import logger

from src.server.notifier import Notifier
from src.server.session import Event, SessionStore, Watch
from src.server.telegram.bot import Bot


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
            await self.bot.send_photo(chat_id, event.image, event.text)
        except httpx.HTTPError as e:
            logger.warning("session={} telegram send failed: {}", session_id, e)


def bind(store: SessionStore, update: dict) -> Optional[tuple[int, str]]:
    """`/start <session_id>` -> (chat_id, reply). Anything else -> None."""
    message = update.get("message") or {}
    text, chat = message.get("text") or "", message.get("chat") or {}
    if not text.startswith("/start") or "id" not in chat:
        return None
    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        return (
            chat["id"],
            "Open the camera page and scan its QR code to bind this chat.",
        )
    try:
        session = store.get(parts[1].strip())
    except KeyError:
        return (
            chat["id"],
            "That session is gone. Reload the camera page and scan again.",
        )
    session.chat_id = chat["id"]
    return chat["id"], f"Bound. I will send a photo when: {session.watch.rule}"


async def poll(bot: Bot, store: SessionStore) -> None:
    """Background loop: long-poll updates, bind chats. Runs until cancelled."""
    # ponytail: long-polling, one instance only (Telegram rejects two pollers) - switch
    # to a webhook if the service ever scales past one process
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
            bound = bind(store, update)
            if bound:
                await bot.send_message(*bound)
