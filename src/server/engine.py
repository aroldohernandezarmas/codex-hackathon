"""One frame through the loop: gate -> model -> tracker -> notifier.

Multi-user: many browser tabs post frames into one event loop. Two rules keep one user
from stalling the others: decode + gate (cv2/numpy, CPU-bound) run in a worker thread,
and the model call runs as a background task - POST /frame returns at once with
sent=True, and the answer shows up in the next frame's status (state/evidence/fired/events).
"""

import asyncio
from datetime import datetime, timezone

from loguru import logger

from src.server.cv.gate import GateResult, decode
from src.server.cv.perception import Perception, PerceptionError
from src.server.notifier import Notifier
from src.server.session import Event, Session, Watch


def _status(session: Session, gate: GateResult, sent: bool) -> dict:
    w = session.watch
    fired, w.fired = w.fired, False  # reported once, on the next status
    return {
        "gate": gate.verdict,
        "streak": gate.streak,
        "sent": sent,
        "busy": session.busy,
        "state": w.tracker.state,
        "evidence": w.evidence,
        "fired": fired,
        "events": len(w.events),
        "usage": session.usage.as_dict(),
    }


async def handle_frame(
    session: Session, jpeg: bytes, perception: Perception, notifier: Notifier
) -> dict:
    async with session.lock:  # gate state is per-session and not thread-safe
        frame = await asyncio.to_thread(decode, jpeg)  # raises ValueError on junk
        gate = await asyncio.to_thread(session.gate.observe, frame)
        skip = "busy" if session.busy else None if gate.send else "gate"
        logger.debug(
            "session={} frame gate={} streak={} {}",
            session.id,
            gate.verdict,
            gate.streak,
            f"skipped: {skip}" if skip else "-> model",
        )
        if skip:
            return _status(session, gate, sent=False)
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
        # ponytail: one predicate per call; for several watches ask them all in one
        # prompt (JSON list) rather than N round-trips
        await _observe(session, jpeg, perception, notifier)
    finally:
        session.busy = False


async def _observe(
    session: Session, jpeg: bytes, perception: Perception, notifier: Notifier
) -> None:
    session_id, w = session.id, session.watch
    try:
        observation = await perception.detect(jpeg, w.predicate)
    except PerceptionError as e:
        logger.warning("session={} perception failed: {}", session_id, e)
        return
    session.usage += observation.usage
    logger.info(
        "session={} usage +{} -> total {}", session_id, observation.usage, session.usage
    )
    w.evidence = observation.evidence
    fired = w.tracker.update(observation.state)
    logger.info(
        "session={} model={} evidence={!r} tracker[{}] fired={}",
        session_id,
        observation.state,
        observation.evidence,
        w.tracker,
        fired,
    )
    if fired:
        became = "true" if w.direction == "rising" else "false"
        event = Event(
            len(w.events),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            f"{w.predicate} - became {became}",
            jpeg,
        )
        w.events.append(event)
        w.fired = True
        await notifier.notify(session_id, w, event)
