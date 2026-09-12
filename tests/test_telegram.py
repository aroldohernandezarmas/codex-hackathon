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
    s = store.create(["the cat <jumps> onto the table"], [RULE])
    s.subscriber = sub.token
    s.watches[0].evidence = "cat on chair"
    r = handle(store, subs, press("status"))
    assert (
        r.callback_id == "cb1"
        and "&lt;jumps&gt;" in r.text  # html-escaped
        and "⏳ still looking" in r.text
        and "cat on chair" in r.text
    )
    assert handle(store, subs, press("stop")).text.startswith("🔕")
    assert sub.chat_id == 42 and sub.muted
    handle(store, subs, press("disconnect"))
    assert sub.chat_id == 42
    handle(store, subs, press("confirm_disconnect"))
    assert sub.chat_id is None
    assert handle(store, subs, press("status")).text.startswith("🔗")


async def test_notify_sends_photo_only_when_bound():
    calls: list = []
    bot = make_bot(calls)
    store = SessionStore(max_sessions=2, ttl=30)
    subs = Subscribers(max_subscribers=4, ttl=3600)
    s = store.create(["x happens"], [RULE])
    event = Event(
        0, "2026-09-12T14:32:10+00:00", "a cat is on the table - became true", b"jpg"
    )
    notifier = TelegramNotifier(bot, store, subs)

    await notifier.notify(s.id, s.watches[0], event)
    assert calls == []  # no subscriber at all: log only
    sub = subs.create()
    s.subscriber = sub.token
    await notifier.notify(s.id, s.watches[0], event)
    assert calls == []  # subscribed but no chat bound yet: still log only
    sub.chat_id = 42
    await notifier.notify(s.id, s.watches[0], event)
    assert calls[0].url.path.endswith("/sendPhoto")
    body = calls[0].content
    assert (
        b'name="chat_id"\r\n\r\n42' in body
        and b"14:32:10 UTC" in body
        and b"jpg" in body
    )
    assert b"inline_keyboard" in body


def connected():
    store = SessionStore(2, 30)
    subs = Subscribers(4, 3600)
    sub = subs.create()
    sub.chat_id = 42
    session = store.create(["cat arrives"], [RULE])
    session.subscriber = sub.token
    return store, subs, sub, session


async def test_snapshot_waits_for_new_frame_and_acknowledges_button():
    import asyncio
    from unittest.mock import AsyncMock

    from src.server.telegram.notifier import respond

    store, subs, sub, session = connected()
    session.latest_frame = b"old event photo"
    session.frame_received.set()
    bot = AsyncMock()
    task = asyncio.create_task(
        respond(bot, store, subs, AsyncMock(), press("snapshot"))
    )
    await asyncio.sleep(0)
    bot.answer_callback.assert_awaited_once_with("cb1")
    bot.send_photo.assert_not_awaited()
    assert not session.frame_received.is_set()
    session.latest_frame = b"fresh camera frame"
    session.frame_at = "2026-09-12T15:00:01+00:00"
    session.frame_received.set()
    await task
    assert bot.send_photo.call_args.args[1] == b"fresh camera frame"
    assert "15:00:01 UTC" in bot.send_photo.call_args.args[2]


async def test_snapshot_timeout_never_sends_stale_image(monkeypatch):
    from unittest.mock import AsyncMock

    from src.server.telegram import notifier

    store, subs, sub, session = connected()
    session.latest_frame = b"stale"
    monkeypatch.setattr(notifier, "SNAPSHOT_TIMEOUT", 0.001)
    bot = AsyncMock()
    await notifier.respond(bot, store, subs, AsyncMock(), press("snapshot"))
    bot.send_photo.assert_not_awaited()
    assert "isn't sending frames" in bot.send_message.call_args.args[1]


async def test_pause_resume_keeps_connection_and_suppresses_only_alerts():
    store, subs, sub, session = connected()
    calls = []
    bot = make_bot(calls)
    notifier = TelegramNotifier(bot, store, subs)
    event = Event(0, "2026-09-12T14:32:10+00:00", "cat arrived", b"jpg")
    handle(store, subs, press("pause"))
    await notifier.notify(session.id, session.watches[0], event)
    assert not calls and sub.chat_id == 42
    handle(store, subs, press("resume"))
    await notifier.notify(session.id, session.watches[0], event)
    assert len(calls) == 1 and not sub.muted
    await bot.aclose()


async def test_rules_menu_add_edit_drop_preserves_history_and_signals_browser():
    from unittest.mock import AsyncMock

    from src.server.telegram.notifier import respond

    store, subs, sub, session = connected()
    history = [Event(0, "2026-09-12T14:32:10+00:00", "cat arrived", b"jpg", "cat")]
    session.events = history
    perception = AsyncMock()
    perception.normalize.return_value = Rule("door open", "rising", True)
    # Rules screen: one row of edit/drop per watch plus Add.
    r = handle(store, subs, press("rules"))
    assert r.buttons[0] == [
        ("✏️ 01", f"edit:{session.id}:0"),
        ("🗑 01", f"drop:{session.id}:0"),
    ]
    assert r.buttons[1] == [("➕ Add rule", "add")] and not sub.editing
    # Add: the old watch keeps its state, the new one starts fresh.
    session.watches[0].tracker.state = True
    handle(store, subs, press("add"))
    assert sub.editing == "add"
    await respond(AsyncMock(), store, subs, perception, msg("The door opens"))
    assert [w.rule for w in session.watches] == ["cat arrives", "The door opens"]
    assert session.watches[0].tracker.state is True
    assert session.watches[1].tracker.state is None
    assert session.retry and session.changed.is_set() and session.revision == 1
    assert not sub.editing and session.events is history
    # Edit replaces in place; the other watch is untouched.
    handle(store, subs, press(f"edit:{session.id}:0"))
    assert sub.editing == "edit:0"
    await respond(AsyncMock(), store, subs, perception, msg("Cat leaves"))
    assert [w.rule for w in session.watches] == ["Cat leaves", "The door opens"]
    assert session.watches[0].tracker.state is None and session.revision == 2
    # Drop: never below one rule.
    handle(store, subs, press(f"drop:{session.id}:1"))
    assert [w.rule for w in session.watches] == ["Cat leaves"] and session.revision == 3
    assert (
        "Keep at least one" in handle(store, subs, press(f"drop:{session.id}:0")).text
    )
    assert len(session.watches) == 1
    assert handle(store, subs, press(f"edit:{session.id}:5")).text.startswith("⌛")


async def test_invalid_rule_retry_and_cancel():
    from unittest.mock import AsyncMock

    from src.server.cv.perception import PerceptionError
    from src.server.telegram.notifier import respond

    store, subs, sub, session = connected()
    old = list(session.watches)
    perception, bot = AsyncMock(), AsyncMock()
    perception.normalize.return_value = Rule("cat", "rising", False)
    handle(store, subs, press("add"))
    await respond(bot, store, subs, perception, msg("cat"))
    assert session.watches == old and sub.editing
    perception.normalize.side_effect = PerceptionError("unavailable")
    await respond(bot, store, subs, perception, msg("door opens"))
    assert session.watches == old and sub.editing
    handle(store, subs, press("cancel"))
    await respond(bot, store, subs, perception, msg("door opens"))
    assert perception.normalize.await_count == 2 and not sub.editing


async def test_menu_edits_existing_text_and_event_access_is_scoped():
    from unittest.mock import AsyncMock

    from src.server.telegram.notifier import respond

    store, subs, sub, session = connected()
    session.events.append(Event(0, "2026-09-12T14:32:10+00:00", "<cat>", b"jpg"))
    bot = AsyncMock()
    update = press("menu")
    update["callback_query"]["message"].update(message_id=10, text="Dashboard")
    await respond(bot, store, subs, AsyncMock(), update)
    bot.edit_message.assert_awaited_once()
    bot.send_message.assert_not_awaited()
    assert handle(store, subs, press(f"event:{session.id}:0")).image == b"jpg"
    assert "&lt;cat&gt;" in handle(store, subs, press(f"event:{session.id}:0")).text
    assert handle(store, subs, press("event:other:0")).image is None
    assert handle(store, subs, press(f"event:{session.id}:-1")).image is None
    assert handle(store, subs, press(f"event:{session.id}:0", 99)).image is None


async def test_group_cannot_bind_or_control_camera():
    from unittest.mock import AsyncMock

    from src.server.telegram.notifier import respond

    store, subs, sub, session = connected()
    update = msg(f"/start {sub.token}", -1)
    update["message"]["chat"]["type"] = "group"
    handle(store, subs, update)
    assert sub.chat_id == 42
    update = msg("/rules door opens")
    update["message"]["chat"]["type"] = "group"
    perception = AsyncMock()
    await respond(AsyncMock(), store, subs, perception, update)
    perception.normalize.assert_not_awaited()


async def test_edit_message_handles_unchanged_screen_but_preserves_api_errors():
    import json

    import pytest

    requests = []
    description = "Bad Request: message is not modified"

    async def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(400, json={"ok": False, "description": description})

    bot = Bot("token")
    await bot.client.aclose()
    bot.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.telegram.org/bottoken",
    )
    await bot.edit_message(42, 10, "Dashboard", [[("Refresh", "menu")]])
    assert requests[0]["message_id"] == 10
    assert (
        requests[0]["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "menu"
    )
    description = "Bad Request: message can't be edited"
    with pytest.raises(httpx.HTTPStatusError):
        await bot.edit_message(42, 10, "Dashboard")
    await bot.aclose()


def test_navigating_away_cancels_rule_entry():
    store, subs, sub, session = connected()
    handle(store, subs, press("add"))
    assert sub.editing
    handle(store, subs, press("events"))
    assert not sub.editing
