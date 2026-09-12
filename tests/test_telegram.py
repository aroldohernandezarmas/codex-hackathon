import httpx

from src.server.cv.perception import Rule
from src.server.session import Event, SessionStore, Subscribers
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
    subs = Subscribers(max_subscribers=4, ttl=3600)
    sub = subs.create()
    assert bot.deep_link(sub.token) == f"https://t.me/cam_bot?start={sub.token}"
    svg = bot.qr_svg(sub.token)
    assert svg.startswith(b"<?xml") and b"<svg" in svg and b"<path" in svg

    assert handle(store, subs, msg("hello")) is None
    assert handle(store, subs, msg("/start")).text.startswith("👋")
    assert handle(store, subs, msg("/start nope")).text.startswith("⌛")
    assert handle(store, subs, msg("/status")).text.startswith("🔗")  # not linked yet

    # Binding happens before any rule exists - that is the point of the subscriber.
    r = handle(store, subs, msg(f"/start {sub.token}"))
    assert (r.chat_id, sub.chat_id) == (42, 42)
    assert r.buttons and handle(store, subs, press("status")).text.startswith("📷")

    # The watch this browser starts later is picked up without a second scan.
    s = store.create("the cat <jumps> onto the table", RULE)
    s.subscriber = sub.token
    s.watch.evidence = "cat on chair"
    r = handle(store, subs, press("status"))
    assert (
        r.callback_id == "cb1"
        and "&lt;jumps&gt;" in r.text  # html-escaped
        and "⏳ still looking" in r.text
        and "cat on chair" in r.text
    )
    assert handle(store, subs, press("stop")).text.startswith("🔕")
    assert sub.chat_id is None
    assert handle(store, subs, press("status")).text.startswith("🔗")


async def test_notify_sends_photo_only_when_bound():
    calls: list = []
    bot = make_bot(calls)
    store = SessionStore(max_sessions=2, ttl=30)
    subs = Subscribers(max_subscribers=4, ttl=3600)
    s = store.create("x happens", RULE)
    event = Event(
        0, "2026-09-12T14:32:10+00:00", "a cat is on the table - became true", b"jpg"
    )
    notifier = TelegramNotifier(bot, store, subs)

    await notifier.notify(s.id, s.watch, event)
    assert calls == []  # no subscriber at all: log only
    sub = subs.create()
    s.subscriber = sub.token
    await notifier.notify(s.id, s.watch, event)
    assert calls == []  # subscribed but no chat bound yet: still log only
    sub.chat_id = 42
    await notifier.notify(s.id, s.watch, event)
    assert calls[0].url.path.endswith("/sendPhoto")
    body = calls[0].content
    assert (
        b'name="chat_id"\r\n\r\n42' in body
        and b"14:32:10 UTC" in body
        and b"jpg" in body
    )
    assert b"inline_keyboard" in body
