from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.server.app import create_app
from src.server.cv.perception import Observation, Rule, Usage
from src.server.notifier import Notifier
from src.server.session import SessionStore, Subscribers
from src.server.telegram.bot import Bot

DATA = Path(__file__).resolve().parents[1] / "data"


class FakePerception:
    def __init__(self) -> None:
        self.answers: list[bool] = []
        self.calls = 0
        self.usage = (0, 0)  # (prompt, completion) billed per detect

    async def normalize(self, rule: str) -> Rule:
        if rule == "a cat":
            return Rule("a cat is visible", "rising", False)
        direction = "falling" if "leaves" in rule else "rising"
        return Rule("a cat is on the table", direction, True)

    async def detect(self, jpeg: bytes, predicate: str) -> Observation:
        self.calls += 1
        return Observation(self.answers.pop(0), "fake", Usage(*self.usage, 1))


class SpyNotifier(Notifier):
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def notify(self, session_id, watch, event) -> None:
        self.sent.append(event.text)


def jpeg(name: str, black_tile: bool = False) -> bytes:
    bgr = cv2.resize(cv2.imread(str(DATA / name)), (1280, 720))
    bgr = (30 + bgr.astype(np.float32) * 0.7).astype(np.uint8)
    if black_tile:
        bgr[:90, :160] = 0
    return cv2.imencode(".jpg", bgr)[1].tobytes()


@pytest.fixture
def world():
    perception, notifier = FakePerception(), SpyNotifier()
    app = create_app(perception, SessionStore(max_sessions=2, ttl=30), notifier)
    return TestClient(app), perception, notifier


def fake_bot() -> Bot:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True, "result": {"username": "cam_bot"}})

    bot = Bot("token")
    bot.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.telegram.org/bottoken",
    )
    # TestClient is not used as a context manager here, so the lifespan - and with it
    # get_me(), which normally fills this in - never runs.
    bot.username = "cam_bot"
    return bot


@pytest.fixture
def bot_world():
    perception, store = FakePerception(), SessionStore(max_sessions=2, ttl=30)
    subs = Subscribers(max_subscribers=4, ttl=3600)
    app = create_app(perception, store, SpyNotifier(), fake_bot(), subs)
    return TestClient(app), store, subs


def post_frame(client, sid, data):
    files = {"frame": ("f.jpg", data, "image/jpeg")}
    return client.post(f"/session/{sid}/frame", files=files)


def new_session(client, rule="the cat jumps onto the table") -> str:
    return client.post("/session", json={"rule": rule}).json()["session_id"]


def test_health_create_reject_and_cap(world):
    client, _, _ = world
    assert client.get("/health").json() == {"ok": True, "sessions": 0}
    r = client.post("/session", json={"rule": "the cat jumps onto the table"})
    assert r.status_code == 201
    assert (r.json()["predicate"], r.json()["direction"]) == (
        "a cat is on the table",
        "rising",
    )
    r = client.post("/session", json={"rule": "a cat"})
    assert (r.status_code, r.json()["error"]) == (400, "not_a_transition")
    new_session(client)
    r = client.post("/session", json={"rule": "x happens"})
    assert (r.status_code, r.json()["error"]) == (503, "full")


def test_frame_flow_fires_once(world):
    client, perception, notifier = world
    sid = new_session(client)
    perception.answers = [False, True, True]
    quiet, changed = jpeg("1.png"), jpeg("1.png", black_tile=True)

    # The model answers in a background task, so each answer shows on the NEXT status.
    s = post_frame(client, sid, quiet).json()  # first frame: sent, baseline
    assert (s["gate"], s["sent"], s["state"]) == ("first", True, None)
    s = post_frame(client, sid, quiet).json()  # same picture: skipped; baseline False
    assert (s["gate"], s["sent"], s["state"]) == ("skip", False, False)
    assert post_frame(client, sid, changed).json()["sent"] is True  # -> True: fires
    s = post_frame(client, sid, changed).json()
    assert (s["fired"], s["state"], s["events"]) == (True, True, 1)
    s = post_frame(client, sid, changed).json()  # == anchor: skip; fired reported once
    assert (s["gate"], s["fired"]) == ("skip", False)
    assert notifier.sent == ["a cat is on the table - became true"]
    assert perception.calls == 2

    view = client.get(f"/session/{sid}").json()
    assert view["events"][0]["n"] == 0 and view["state"] is True
    img = client.get(f"/session/{sid}/events/0.jpg")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    ev = client.get(f"/session/{sid}/events/0").json()
    assert (ev["n"], ev["text"]) == (0, view["events"][0]["text"])
    assert ev["image"].startswith("data:image/jpeg;base64,")
    assert client.get(f"/session/{sid}/events/1").status_code == 404
    assert view["telegram"] is False  # nothing subscribed


def test_subscribe_before_any_watch_then_link(bot_world):
    """The page can offer a QR with no session running, and one scan covers later watches."""
    client, store, subs = bot_world
    r = client.post("/subscriber")
    assert r.status_code == 201
    sub = r.json()
    token = sub["token"]
    assert sub["telegram_link"] == f"https://t.me/cam_bot?start={token}"
    assert sub["linked"] is False

    qr = client.get(f"/subscriber/{token}/qr.svg")
    assert (qr.status_code, qr.headers["content-type"]) == (200, "image/svg+xml")
    assert qr.content.startswith(b"<?xml") and b"<path" in qr.content

    # The page polls this; it is the only thing driving the QR's visibility.
    assert client.get(f"/subscriber/{token}").json()["linked"] is False
    subs.get(token).chat_id = 42  # what /start <token> does in the bot
    assert client.get(f"/subscriber/{token}").json()["linked"] is True

    # A watch started afterwards inherits the binding - no second scan.
    sid = client.post(
        "/session", json={"rule": "the cat jumps onto the table", "subscriber": token}
    ).json()["session_id"]
    assert store.get(sid).subscriber == token
    assert client.get(f"/session/{sid}").json()["telegram"] is True


def test_unknown_subscriber_is_404(bot_world):
    """A token from a previous process must fail cleanly so the page makes a new one."""
    client, store, _ = bot_world
    assert client.get("/subscriber/nope").status_code == 404
    assert client.get("/subscriber/nope/qr.svg").status_code == 404
    # An unknown token on /session is ignored rather than rejected: the watch still runs.
    sid = client.post(
        "/session", json={"rule": "the cat jumps onto the table", "subscriber": "nope"}
    ).json()["session_id"]
    assert store.get(sid).subscriber is None
    assert client.get(f"/session/{sid}").json()["telegram"] is False


def test_errors_and_delete(world):
    client, _, _ = world
    assert post_frame(client, "nope", jpeg("1.png")).status_code == 404
    assert client.get("/session/nope").status_code == 404
    sid = new_session(client, "x happens")
    assert post_frame(client, sid, b"not a jpeg").status_code == 400
    assert client.delete(f"/session/{sid}").status_code == 204
    assert client.get(f"/session/{sid}").status_code == 404


def test_usage_accumulates_per_session_and_resets(world):
    client, perception, _ = world
    perception.usage = (400, 10)
    sid = new_session(client)
    assert client.get(f"/session/{sid}/usage").json()["total"] == 0
    perception.answers = [True]
    post_frame(client, sid, jpeg("1.png"))
    assert client.get(f"/session/{sid}/usage").json() == {
        "prompt": 400,
        "completion": 10,
        "total": 410,
        "calls": 1,
        "usd": 0.00135,  # 400*$3 + 10*$15 per 1M
    }
    other = new_session(client)
    assert client.get(f"/session/{other}/usage").json()["total"] == 0
    assert client.post(f"/session/{sid}/usage/reset").json()["total"] == 0
    assert client.get(f"/session/{sid}").json()["usage"]["calls"] == 0


def test_empty_frame_and_negative_event_numbers(world):
    client, perception, _ = world
    sid = new_session(client)
    assert post_frame(client, sid, b"").status_code == 400
    for suffix in ("-1", "-1.jpg", "-999"):
        assert client.get(f"/session/{sid}/events/{suffix}").status_code == 404
    perception.answers = [True]
    post_frame(client, sid, jpeg("1.png"))
    assert client.get(f"/session/{sid}/events/0").status_code == 200
    assert client.get(f"/session/{sid}/events/-1").status_code == 404


async def test_capacity_reserved_before_normalizing_and_released_on_failure():
    import asyncio

    import httpx

    from src.server.cv.perception import PerceptionError

    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    class SlowPerception(FakePerception):
        async def normalize(self, rule):
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            if calls == 1:
                raise PerceptionError("temporary failure")
            return await super().normalize(rule)

    app = create_app(SlowPerception(), SessionStore(1, 30), Notifier())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = asyncio.create_task(client.post("/session", json={"rule": "arrives"}))
        await entered.wait()
        assert (
            await client.post("/session", json={"rule": "arrives"})
        ).status_code == 503
        assert calls == 1
        release.set()
        assert (await first).status_code == 502
        assert (
            await client.post("/session", json={"rule": "arrives"})
        ).status_code == 201
        assert (
            await client.post("/session", json={"rule": "arrives"})
        ).status_code == 503
        assert calls == 2


async def test_lifespan_expires_sessions_without_requests():
    import asyncio

    store = SessionStore(1, 30)
    s = store.create("arrives", Rule("present", "rising", True))
    app = create_app(FakePerception(), store, Notifier())
    async with app.router.lifespan_context(app):
        s.last_seen = 0
        await asyncio.sleep(0)
        assert not store._sessions


async def test_sse_delivers_result_without_another_frame_and_closes_on_delete():
    import asyncio
    import json

    from src.server.engine import handle_frame

    store = SessionStore(1, 30)
    perception = FakePerception()
    perception.answers = [True]
    s = store.create("arrives", Rule("present", "rising", True))
    app = create_app(perception, store, Notifier())
    endpoint = next(
        r.endpoint
        for r in app.routes
        if getattr(r, "path", "") == "/session/{session_id}/updates"
    )
    response = await endpoint(s.id)
    stream = response.body_iterator
    assert response.media_type == "text/event-stream"
    assert json.loads((await anext(stream)).removeprefix("data: "))["state"] is None
    pending = asyncio.create_task(anext(stream))
    await handle_frame(s, jpeg("1.png"), perception, Notifier())
    await s.task
    status = json.loads((await asyncio.wait_for(pending, 1)).removeprefix("data: "))
    assert status["state"] is True and status["events"] == 1
    assert status["revision"] == 1
    pending = asyncio.create_task(anext(stream))
    store.delete(s.id)
    assert "event: expired" in await asyncio.wait_for(pending, 1)
    await stream.aclose()
