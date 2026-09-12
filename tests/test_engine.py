import asyncio

import cv2
import numpy as np
import pytest

from src.server.cv.perception import Detection, Observation, PerceptionError, Rule
from src.server.engine import handle_frame
from src.server.notifier import Notifier
from src.server.session import SessionStore


def image(changed=False):
    pixels = np.full((720, 1280, 3), 150, np.uint8)
    if changed:
        pixels[:90, :160] = 0
    return cv2.imencode(".jpg", pixels)[1].tobytes()


async def test_change_while_busy_is_detected_when_model_finishes():
    release = asyncio.Event()
    calls = []

    class Perception:
        async def detect(self, jpeg, predicate):
            calls.append(jpeg)
            if len(calls) == 1:
                await release.wait()
            return Detection([Observation(len(calls) > 1, "observed")])

    s = SessionStore(1, 30).create(["arrives"], [Rule("present", "rising", True)])
    p, notifier = Perception(), Notifier()
    await handle_frame(s, image(), p, notifier)
    await handle_frame(s, image(True), p, notifier)
    status = await handle_frame(s, image(True), p, notifier)
    assert status["busy"] and not status["sent"]
    release.set()
    await s.task
    assert (await handle_frame(s, image(True), p, notifier))["sent"]
    await s.task
    assert calls == [image(), image(True)]
    assert len(s.events) == 1


@pytest.mark.parametrize("fail_call", [1, 2])
async def test_failed_perception_retries_unchanged_scene(fail_call):
    calls = 0

    class Perception:
        async def detect(self, jpeg, predicate):
            nonlocal calls
            calls += 1
            if calls == fail_call:
                raise PerceptionError("temporary failure")
            return Detection([Observation(calls > 1, "observed")])

    s = SessionStore(1, 30).create(["arrives"], [Rule("present", "rising", True)])
    p, notifier = Perception(), Notifier()
    await handle_frame(s, image(), p, notifier)
    await s.task
    frame = image(fail_call == 2)
    if fail_call == 2:
        assert (await handle_frame(s, frame, p, notifier))["sent"]
        await s.task
    assert s.retry
    assert (await handle_frame(s, frame, p, notifier))["sent"]
    await s.task
    assert not s.retry and calls == fail_call + 1
    assert s.watches[0].tracker.state is True


async def test_slow_notification_does_not_block_detection_and_keeps_evidence():
    release = asyncio.Event()
    started = asyncio.Event()
    captions = []

    class Perception:
        calls = 0

        async def detect(self, jpeg, predicate):
            self.calls += 1
            return Detection([Observation(self.calls == 1, f"answer {self.calls}")])

    class SlowNotifier:
        async def notify(self, sid, watch, event):
            started.set()
            await release.wait()
            captions.append(watch.evidence)

    store = SessionStore(1, 30)
    s = store.create(["arrives"], [Rule("present", "rising", True)])
    p, notifier = Perception(), SlowNotifier()
    await handle_frame(s, image(), p, notifier)
    await s.task
    await started.wait()
    assert not s.busy and s.changed.is_set()
    assert (await handle_frame(s, image(True), p, notifier))["sent"]
    await s.task
    assert p.calls == 2 and s.watches[0].evidence == "answer 2"
    release.set()
    await asyncio.gather(*s.notifications)
    assert captions == ["answer 1"]


async def test_latest_frame_updates_even_when_model_is_busy():
    from unittest.mock import AsyncMock

    s = SessionStore(1, 30).create(["arrives"], [Rule("present", "rising", True)])
    s.busy = True
    jpeg = image()
    await handle_frame(s, jpeg, AsyncMock(), Notifier())
    assert s.latest_frame == jpeg and s.frame_at and s.frame_received.is_set()
    s.frame_received.clear()
    with pytest.raises(ValueError):
        await handle_frame(s, b"invalid", AsyncMock(), Notifier())
    assert not s.frame_received.is_set() and s.latest_frame == jpeg


async def test_old_detection_cannot_fire_after_rule_change():
    from src.server.engine import perceive
    from src.server.session import new_watch

    started, release = asyncio.Event(), asyncio.Event()

    class SlowPerception:
        async def detect(self, jpeg, predicate):
            started.set()
            await release.wait()
            return Detection([Observation(True, "old scene")])

    s = SessionStore(1, 30).create(["arrives"], [Rule("present", "rising", True)])
    task = asyncio.create_task(perceive(s, b"jpg", SlowPerception(), Notifier()))
    await started.wait()
    s.watches[0] = new_watch("door opens", Rule("door open", "rising", True))
    release.set()
    await task
    assert not s.events and not s.watches[0].evidence
    assert s.watches[0].tracker.state is None


async def test_several_watches_share_one_call_and_fire_independently():
    asked = []

    class Perception:
        async def detect(self, jpeg, predicates):
            asked.append(predicates)
            return Detection(
                [Observation(True, "cat here"), Observation(len(asked) > 1, "door")]
            )

    s = SessionStore(1, 30).create(
        ["cat arrives", "door opens"],
        [Rule("cat present", "rising", True), Rule("door open", "rising", True)],
    )
    p, notifier = Perception(), Notifier()
    await handle_frame(s, image(), p, notifier)
    await s.task
    assert asked == [["cat present", "door open"]]
    assert [e.rule for e in s.events] == ["cat arrives"]
    status = await handle_frame(s, image(True), p, notifier)
    assert status["fired"] and status["watches"][1]["state"] is False
    await s.task
    assert [e.rule for e in s.events] == ["cat arrives", "door opens"]
    assert [e.n for e in s.events] == [0, 1]
    assert (await handle_frame(s, image(True), p, notifier))["fired"]
