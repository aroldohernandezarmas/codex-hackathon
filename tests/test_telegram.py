import json

import httpx

from src.server.cv.perception import Rule
from src.server.session import Event, SessionStore
from src.server.telegram.bot import Bot
from src.server.telegram.notifier import TelegramNotifier, bind

RULE = Rule("a cat is on the table", "rising", True)


def make_bot(calls: list) -> Bot:
    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"ok": True, "result": {"username": "cam_bot"}})

    bot = Bot("token")
    bot.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.telegram.org/bottoken",
    )
    return bot


def start(text: str, chat_id: int = 42) -> dict:
    return {"update_id": 1, "message": {"text": text, "chat": {"id": chat_id}}}


async def test_deep_link_and_bind():
    calls: list = []
    bot = make_bot(calls)
    assert await bot.get_me() == "cam_bot"
    store = SessionStore(max_sessions=2, ttl=30)
    s = store.create("the cat jumps onto the table", RULE)
    assert bot.deep_link(s.id) == f"https://t.me/cam_bot?start={s.id}"

    assert bind(store, start("/start nope"))[1].startswith("That session is gone")
    assert bind(store, {"message": {"text": "hello", "chat": {"id": 1}}}) is None
    chat_id, reply = bind(store, start(f"/start {s.id}"))
    assert (chat_id, s.chat_id) == (42, 42)
    assert reply.endswith("the cat jumps onto the table")


async def test_notify_sends_photo_only_when_bound():
    calls: list = []
    bot = make_bot(calls)
    store = SessionStore(max_sessions=2, ttl=30)
    s = store.create("x happens", RULE)
    event = Event(
        0, "2026-09-12T14:32:10Z", "a cat is on the table - became true", b"jpg"
    )
    notifier = TelegramNotifier(bot, store)

    await notifier.notify(s.id, s.watch, event)
    assert calls == []  # not bound: log only
    s.chat_id = 42
    await notifier.notify(s.id, s.watch, event)
    assert calls[0].url.path.endswith("/sendPhoto")
    body = calls[0].content
    assert (
        b'name="chat_id"\r\n\r\n42' in body
        and b"became true" in body
        and b"jpg" in body
    )
