import httpx

from src.server.cv.perception import Rule
from src.server.session import Event, SessionStore
from src.server.telegram.bot import Bot
from src.server.telegram.notifier import TelegramNotifier, handle

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


def msg(text: str, chat_id: int = 42) -> dict:
    return {"update_id": 1, "message": {"text": text, "chat": {"id": chat_id}}}


def press(data: str, chat_id: int = 42) -> dict:
    return {
        "update_id": 2,
        "callback_query": {
            "id": "cb1",
            "data": data,
            "message": {"chat": {"id": chat_id}},
        },
    }


async def test_deep_link_bind_status_stop():
    bot = make_bot([])
    assert await bot.get_me() == "cam_bot"
    store = SessionStore(max_sessions=2, ttl=30)
    s = store.create("the cat <jumps> onto the table", RULE)
    assert bot.deep_link(s.id) == f"https://t.me/cam_bot?start={s.id}"
    svg = bot.qr_svg(s.id)
    assert svg.startswith(b"<?xml") and b"<svg" in svg and b"<path" in svg

    assert handle(store, msg("hello")) is None
    assert handle(store, msg("/start")).text.startswith("👋")
    assert handle(store, msg("/start nope")).text.startswith("⌛")
    assert handle(store, msg("/status")).text.startswith("🔗")  # not linked yet
    r = handle(store, msg(f"/start {s.id}"))
    assert (r.chat_id, s.chat_id) == (42, 42)
    assert "&lt;jumps&gt;" in r.text and r.buttons  # html-escaped, with buttons

    s.watch.evidence = "cat on chair"
    r = handle(store, press("status"))
    assert (
        r.callback_id == "cb1"
        and "⏳ still looking" in r.text
        and "cat on chair" in r.text
    )
    assert handle(store, press("stop")).text.startswith("🔕")
    assert s.chat_id is None and handle(store, press("status")).text.startswith("🔗")


async def test_notify_sends_photo_only_when_bound():
    calls: list = []
    bot = make_bot(calls)
    store = SessionStore(max_sessions=2, ttl=30)
    s = store.create("x happens", RULE)
    event = Event(
        0, "2026-09-12T14:32:10+00:00", "a cat is on the table - became true", b"jpg"
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
        and b"14:32:10 UTC" in body
        and b"jpg" in body
    )
    assert b"inline_keyboard" in body
