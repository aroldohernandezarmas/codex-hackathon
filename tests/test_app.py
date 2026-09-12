from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.server.app import create_app
from src.server.cv.perception import Observation, Rule, Usage
from src.server.notifier import Notifier
from src.server.session import SessionStore

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
    assert client.get(f"/session/{sid}/qr.svg").status_code == 404  # no bot in tests


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
    }
    other = new_session(client)
    assert client.get(f"/session/{other}/usage").json()["total"] == 0
    assert client.post(f"/session/{sid}/usage/reset").json()["total"] == 0
    assert client.get(f"/session/{sid}").json()["usage"]["calls"] == 0
