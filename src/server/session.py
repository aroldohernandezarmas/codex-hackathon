"""In-memory sessions: one gate + one tracker per browser tab, capped, expiring on silence."""

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional

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
class Subscriber:
    """A browser that asked for Telegram alerts.

    Deliberately not part of Session: the chat is bound once, before any rule exists,
    and carries into every watch that browser starts afterwards."""

    token: str
    chat_id: Optional[int] = None  # Telegram chat bound via /start <token>
    last_seen: float = field(default_factory=time.monotonic)


@dataclass
class Session:
    id: str
    gate: Gate
    watch: Watch  # ponytail: one rule per session; -> watches: list[Watch] for several
    busy: bool = False  # a model call is in flight
    subscriber: Optional[str] = None  # Subscriber.token to notify when an event fires
    chat_id: Optional[int] = None  # Telegram chat bound via /start <id>
    usage: Usage = field(default_factory=Usage)  # API tokens spent by this session
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
        session.usage += spec.usage  # the normalize call is billed to this session
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


class Subscribers:
    """Browser -> Telegram chat, keyed by an opaque token the page keeps in localStorage.

    Outlives sessions, dies with the process - the same "nothing is stored between
    restarts" rule the sessions follow. A restart just means the page subscribes again.
    """

    def __init__(self, max_subscribers: int, ttl: float) -> None:
        self.max_subscribers = max_subscribers
        self.ttl = ttl
        self._subscribers: dict[str, Subscriber] = {}

    def __len__(self) -> int:
        return len(self._subscribers)

    def create(self) -> Subscriber:
        self.sweep()
        # Evict the stalest instead of refusing: subscribing costs nothing and a new
        # browser must always get a QR, unlike a session which holds a model budget.
        while len(self._subscribers) >= self.max_subscribers:
            stalest = min(self._subscribers.values(), key=lambda s: s.last_seen)
            del self._subscribers[stalest.token]
        subscriber = Subscriber(secrets.token_urlsafe(9))
        self._subscribers[subscriber.token] = subscriber
        return subscriber

    def __contains__(self, token: object) -> bool:
        return token in self._subscribers

    def get(self, token: str) -> Subscriber:
        """Raises KeyError. Touches: an open page polls, and polling keeps it alive."""
        subscriber = self._subscribers[token]
        subscriber.last_seen = time.monotonic()
        return subscriber

    def by_chat(self, chat_id: int) -> Optional[Subscriber]:
        # ponytail: linear scan, MAX_SUBSCRIBERS is small
        return next(
            (s for s in self._subscribers.values() if s.chat_id == chat_id), None
        )

    def sweep(self, now: float | None = None) -> int:
        now = time.monotonic() if now is None else now
        dead = [k for k, s in self._subscribers.items() if now - s.last_seen > self.ttl]
        for k in dead:
            del self._subscribers[k]
        return len(dead)
