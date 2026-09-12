"""HTTP surface. The contract lives in docs/superpowers/specs/2026-09-12-camera-events-design.md."""

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from src import config
from src.server.cv.perception import GrokPerception, Perception, PerceptionError, Usage
from src.server.engine import detection_status, handle_frame
from src.server.notifier import Notifier
from src.server.session import Session, SessionFull, SessionStore
from src.server.telegram.bot import Bot
from src.server.telegram.notifier import TelegramNotifier, poll

STATIC = Path(__file__).resolve().parents[2] / "static"


class NewSession(BaseModel):
    rule: str


def create_app(
    perception: Perception,
    store: SessionStore,
    notifier: Notifier,
    bot: Optional[Bot] = None,
) -> FastAPI:
    pending_sessions = 0

    async def expire_sessions():
        while True:
            store.sweep()
            await asyncio.sleep(1)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        task = None
        if bot is not None:
            await bot.get_me()
            task = asyncio.create_task(poll(bot, store))
        sweeper = asyncio.create_task(expire_sessions())
        try:
            yield
        finally:
            tasks = [sweeper] + ([task] if task is not None else [])
            for session in store.all():
                tasks.extend(session.notifications)
                if session.task is not None:
                    tasks.append(session.task)
                store.delete(session.id)
            for background in tasks:
                background.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    app = FastAPI(title="camera events", lifespan=lifespan)

    def session_or_404(session_id: str) -> Session:
        try:
            return store.get(session_id)
        except KeyError:
            raise HTTPException(404, "no such session")

    def event_or_404(session_id: str, n: int):
        s = session_or_404(session_id)
        if n < 0 or n >= len(s.watch.events):
            raise HTTPException(404, "no such event")
        return s.watch.events[n]

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    async def health():
        return {"ok": True, "sessions": len(store)}

    @app.post("/session", status_code=201)
    async def create_session(body: NewSession):
        nonlocal pending_sessions
        rule = body.rule.strip()
        if not rule:
            return JSONResponse(
                {"error": "empty_rule", "hint": "describe something that happens"},
                status_code=400,
            )
        if len(store) + pending_sessions >= store.max_sessions:
            return JSONResponse({"error": "full"}, status_code=503)
        pending_sessions += 1
        try:
            spec = await perception.normalize(rule)
            if not spec.is_transition:
                return JSONResponse(
                    {
                        "error": "not_a_transition",
                        "hint": "describe something that happens - e.g. 'the cat jumps onto the table'",
                    },
                    status_code=400,
                )
            session = store.create(rule, spec)
        except PerceptionError as e:
            return JSONResponse(
                {"error": "perception", "hint": str(e)}, status_code=502
            )
        except SessionFull:
            return JSONResponse({"error": "full"}, status_code=503)
        finally:
            pending_sessions -= 1
        w = session.watch
        return {
            "session_id": session.id,
            "predicate": w.predicate,
            "direction": w.direction,
            "telegram_link": bot.deep_link(session.id) if bot else None,
            "usage": session.usage.as_dict(),
        }

    @app.post("/session/{session_id}/frame")
    async def frame(session_id: str, frame: UploadFile = File(...)):
        session = session_or_404(session_id)
        store.touch(session)
        try:
            return await handle_frame(session, await frame.read(), perception, notifier)
        except ValueError:
            raise HTTPException(400, "not a decodable image")

    @app.get("/session/{session_id}")
    async def view(session_id: str):
        s = session_or_404(session_id)
        w = s.watch
        return {
            "session_id": s.id,
            "rule": w.rule,
            "predicate": w.predicate,
            "direction": w.direction,
            "state": w.tracker.state,
            "evidence": w.evidence,
            "telegram": s.chat_id is not None,
            "usage": s.usage.as_dict(),
            "events": [{"n": e.n, "at": e.at, "text": e.text} for e in w.events],
        }

    @app.get("/session/{session_id}/updates")
    async def updates(session_id: str):
        session = session_or_404(session_id)

        async def stream():
            while not session.closed:
                session.changed.clear()
                yield "data: " + json.dumps(detection_status(session)) + "\n\n"
                try:
                    await asyncio.wait_for(session.changed.wait(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
            yield "event: expired\ndata: {}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/session/{session_id}/usage")
    async def usage(session_id: str):
        return session_or_404(session_id).usage.as_dict()

    @app.post("/session/{session_id}/usage/reset")
    async def reset_usage(session_id: str):
        session = session_or_404(session_id)
        logger.info("session={} usage reset from {}", session_id, session.usage)
        session.usage = Usage()
        session.revision += 1
        session.changed.set()
        return session.usage.as_dict()

    @app.get("/session/{session_id}/qr.svg")
    async def qr(session_id: str):
        session = session_or_404(session_id)
        if bot is None:
            raise HTTPException(404, "no telegram bot configured")
        return Response(bot.qr_svg(session.id), media_type="image/svg+xml")

    @app.get(
        "/session/{session_id}/events/{n}.jpg"
    )  # before /{n}: "0.jpg" is not an int
    async def event_image(session_id: str, n: int):
        return Response(event_or_404(session_id, n).image, media_type="image/jpeg")

    @app.get("/session/{session_id}/events/{n}")
    async def event(session_id: str, n: int):
        e = event_or_404(session_id, n)
        return {
            "n": e.n,
            "at": e.at,
            "text": e.text,
            "image": "data:image/jpeg;base64," + base64.b64encode(e.image).decode(),
        }

    @app.delete("/session/{session_id}", status_code=204)
    async def delete(session_id: str):
        store.delete(session_id)
        return Response(status_code=204)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def default_app() -> FastAPI:
    perception = GrokPerception(config.XAI_API_KEYS, config.XAI_MODEL)
    store = SessionStore(config.MAX_SESSIONS, config.SESSION_TTL)
    if not config.TELEGRAM_BOT_TOKEN:
        return create_app(perception, store, Notifier())
    bot = Bot(config.TELEGRAM_BOT_TOKEN)
    return create_app(perception, store, TelegramNotifier(bot, store), bot)
