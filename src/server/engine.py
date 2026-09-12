"""One frame through the loop: gate -> model -> tracker -> notifier.

Multi-user: many browser tabs post frames into one event loop. Two rules keep one user
from stalling the others: decode + gate (cv2/numpy, CPU-bound) run in a worker thread,
and the model call runs as a background task - POST /frame returns at once with
sent=True, and model results are pushed to the browser over SSE.
"""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone

from loguru import logger

from src.server.cv.gate import GateResult, decode
from src.server.cv.perception import Perception, PerceptionError
from src.server.notifier import Notifier
from src.server.session import Event, Session, Watch


def watch_status(w: Watch) -> dict:
    return {
        "rule": w.rule,
        "predicate": w.predicate,
        "direction": w.direction,
        "state": w.tracker.state,
        "evidence": w.evidence,
    }


def detection_status(session: Session) -> dict:
    return {
        "revision": session.revision,
        "watches": [watch_status(w) for w in session.watches],
        "events": len(session.events),
        "last_event": session.events[-1].text if session.events else None,
        "usage": session.usage.as_dict(),
    }


def _status(session: Session, gate: GateResult, sent: bool) -> dict:
    fired = any(w.fired for w in session.watches)
    for w in session.watches:
        w.fired = False  # reported once, on the next status
    return {
        "gate": gate.verdict,
        "streak": gate.streak,
        "sent": sent,
        "busy": session.busy,
        "fired": fired,
        **detection_status(session),
    }


async def handle_frame(
    session: Session, jpeg: bytes, perception: Perception, notifier: Notifier
) -> dict:
    async with session.lock:  # gate state is per-session and not thread-safe
        frame = await asyncio.to_thread(decode, jpeg)  # raises ValueError on junk
        session.latest_frame = jpeg
        session.frame_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        session.frame_received.set()
        gate = await asyncio.to_thread(session.gate.observe, frame, not session.busy)
        skip = (
            "busy" if session.busy else None if gate.send or session.retry else "gate"
        )
        logger.debug(
            "session={} frame gate={} streak={} {}",
            session.id,
            gate.verdict,
            gate.streak,
            f"skipped: {skip}" if skip else "-> model",
        )
        if skip:
            return _status(session, gate, sent=False)
        session.retry = False
        session.busy = True
        session.task = asyncio.create_task(
            perceive(session, jpeg, perception, notifier)
        )
        return _status(session, gate, sent=True)


async def perceive(
    session: Session, jpeg: bytes, perception: Perception, notifier: Notifier
) -> None:
    """Background: ask the model, update the tracker, fire the edge. Never raises."""
    try:
        await _observe(session, jpeg, perception, notifier)
    finally:
        session.busy = False
        session.revision += 1
        session.changed.set()


async def _notify(
    notifier: Notifier, session_id: str, watch: Watch, event: Event
) -> None:
    try:
        await notifier.notify(session_id, watch, event)
    except Exception:
        logger.exception("session={} notification failed", session_id)


async def _observe(
    session: Session, jpeg: bytes, perception: Perception, notifier: Notifier
) -> None:
    session_id, watches = session.id, list(session.watches)  # snapshot: edits race
    try:
        detection = await perception.detect(jpeg, [w.predicate for w in watches])
    except PerceptionError as e:
        session.retry = True
        logger.warning("session={} perception failed: {}", session_id, e)
        return
    session.usage += detection.usage
    logger.info(
        "session={} usage +{} -> total {}", session_id, detection.usage, session.usage
    )
    if session.closed:
        return
    for w, observation in zip(watches, detection.observations):
        if not any(w is live for live in session.watches):
            continue  # replaced or dropped while the model was thinking
        w.evidence = observation.evidence
        fired = w.tracker.update(observation.state)
        logger.info(
            "session={} {!r} model={} evidence={!r} tracker[{}] fired={}",
            session_id,
            w.predicate,
            observation.state,
            observation.evidence,
            w.tracker,
            fired,
        )
        if not fired:
            continue
        became = "true" if w.direction == "rising" else "false"
        event = Event(
            len(session.events),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            f"{w.predicate} - became {became}",
            jpeg,
            w.rule,
        )
        session.events.append(event)
        w.fired = True
        task = asyncio.create_task(_notify(notifier, session_id, replace(w), event))
        session.notifications.add(task)
        task.add_done_callback(session.notifications.discard)
