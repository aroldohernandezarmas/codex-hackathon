from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.server.app import create_app
from src.server.cv.perception import Observation, Rule
from src.server.notifier import Notifier
from src.server.session import SessionStore, Subscribers
from src.server.telegram.bot import Bot

DATA = Path(__file__).resolve().parents[1] / "data"


class FakePerception:
    def __init__(self) -> None:
        self.answers: list[bool] = []
        self.calls = 0

    async def normalize(self, rule: str) -> Rule:
        if rule == "a cat":
            return Rule("a cat is visible", "rising", False)
        direction = "falling" if "leaves" in rule else "rising"
        return Rule("a cat is on the table", direction, True)

    async def detect(self, jpeg: bytes, predicate: str) -> Observation:
        self.calls += 1
        return Observation(self.answers.pop(0), "fake")


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
    perception.answers = [False, False, True, True, True]
    quiet, changed = jpeg("1.png"), jpeg("1.png", black_tile=True)

    # The model answers in a background task, so each answer shows on the NEXT status.
    s = post_frame(client, sid, quiet).json()  # first frame: sent, baseline candidate
    assert (s["gate"], s["sent"], s["state"]) == ("first", True, None)
    s = post_frame(client, sid, quiet).json()  # same picture: skipped; False 1/2
    assert (s["gate"], s["sent"], s["state"]) == ("skip", False, None)
    assert post_frame(client, sid, changed).json()["streak"] == 1  # not yet persisted
    assert post_frame(client, sid, changed).json()["sent"] is True  # -> False 2/2
    s = post_frame(client, sid, quiet).json()  # baseline False confirmed
    assert (s["streak"], s["state"]) == (1, False)
    assert post_frame(client, sid, quiet).json()["sent"] is True  # -> True 1/2
    s = post_frame(client, sid, changed).json()
    assert (s["streak"], s["fired"], s["state"]) == (1, False, False)
    assert post_frame(client, sid, changed).json()["sent"] is True  # -> True 2/2: fires
    s = post_frame(client, sid, quiet).json()
    assert (s["fired"], s["state"], s["events"]) == (True, True, 1)
    s = post_frame(client, sid, changed).json()  # == anchor: skip; fired reported once
    assert (s["gate"], s["fired"]) == ("skip", False)
    assert notifier.sent == ["a cat is on the table - became true"]
    assert perception.calls == 4

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
