"""In-memory sessions: one gate + one tracker per browser tab, capped, expiring on silence."""

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional

from src.server.cv.gate import Gate
from src.server.cv.perception import Rule
from src.server.tracker import Tracker


class SessionFull(Exception):
    pass


@dataclass
class Event:
    n: int
    at: str  # ISO-8601 UTC
    text: str
    image: bytes


@dataclass
class Watch:
    """One rule the user is waiting for. A session has one today; the engine loops
    over watches, so N rules per camera is a list here plus one prompt change."""

    rule: str
    predicate: str
    direction: str
    tracker: Tracker
    evidence: str = ""
    fired: bool = False  # an event fired, not yet reported in a FrameStatus
    events: list[Event] = field(default_factory=list)


@dataclass
class Session:
    id: str
    gate: Gate
    watch: Watch  # ponytail: one rule per session; -> watches: list[Watch] for several
    busy: bool = False  # a model call is in flight
    chat_id: Optional[int] = None  # Telegram chat bound via /start <id>
    last_seen: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(
        default_factory=asyncio.Lock
    )  # frames of one tab, in order
    task: Optional["asyncio.Task[None]"] = None  # keeps the background model call alive


class SessionStore:
    def __init__(self, max_sessions: int, ttl: float) -> None:
        self.max_sessions = max_sessions
        self.ttl = ttl
        self._sessions: dict[str, Session] = {}

    def __len__(self) -> int:
        return len(self._sessions)

    def create(self, rule: str, spec: Rule) -> Session:
        self.sweep()
        if len(self._sessions) >= self.max_sessions:
            raise SessionFull()
        session = Session(
            secrets.token_urlsafe(6),
            Gate(),
            Watch(rule, spec.predicate, spec.direction, Tracker(spec.direction)),
        )
        self._sessions[session.id] = session
        return session

    def all(self) -> list[Session]:
        return list(self._sessions.values())

    def get(self, session_id: str) -> Session:
        return self._sessions[session_id]

    def touch(self, session: Session) -> None:
        session.last_seen = time.monotonic()

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def sweep(self, now: float | None = None) -> int:
        now = time.monotonic() if now is None else now
        dead = [k for k, s in self._sessions.items() if now - s.last_seen > self.ttl]
        for k in dead:
            del self._sessions[k]
        return len(dead)
