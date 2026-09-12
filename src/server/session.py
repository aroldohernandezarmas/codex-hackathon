"""In-memory sessions: one gate + one tracker per browser tab, capped, expiring on silence."""

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from src.server.cv.gate import Gate
from src.server.cv.perception import Rule, Usage
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
    retry: bool = False  # retry failed perception on the next available frame
    chat_id: Optional[int] = None  # Telegram chat bound via /start <id>
    usage: Usage = field(default_factory=Usage)  # API tokens spent by this session
    revision: int = 0
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    closed: bool = False
    notifications: set[asyncio.Task] = field(default_factory=set)
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
        self.sweep()
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
        session.usage += spec.usage  # the normalize call is billed to this session
        logger.info("session={} usage +{} (normalize)", session.id, spec.usage)
        self._sessions[session.id] = session
        return session

    def all(self) -> list[Session]:
        self.sweep()
        return list(self._sessions.values())

    def get(self, session_id: str) -> Session:
        self.sweep()
        return self._sessions[session_id]

    def touch(self, session: Session) -> None:
        session.last_seen = time.monotonic()

    def delete(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.closed = True
            session.changed.set()
            if session.task is not None:
                session.task.cancel()
            for task in session.notifications:
                task.cancel()

    def sweep(self, now: float | None = None) -> int:
        now = time.monotonic() if now is None else now
        dead = [k for k, s in self._sessions.items() if now - s.last_seen > self.ttl]
        for k in dead:
            self.delete(k)
        return len(dead)
