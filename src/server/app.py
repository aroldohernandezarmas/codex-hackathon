"""HTTP surface. The contract lives in docs/superpowers/specs/2026-09-12-camera-events-design.md."""

import base64
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import config
from src.server.cv.perception import GrokPerception, Perception, PerceptionError
from src.server.engine import handle_frame
from src.server.notifier import Notifier
from src.server.session import Session, SessionFull, SessionStore

STATIC = Path(__file__).resolve().parents[2] / "static"


class NewSession(BaseModel):
    rule: str


def create_app(
    perception: Perception, store: SessionStore, notifier: Notifier
) -> FastAPI:
    app = FastAPI(title="camera events")

    def session_or_404(session_id: str) -> Session:
        try:
            return store.get(session_id)
        except KeyError:
            raise HTTPException(404, "no such session")

    def event_or_404(session_id: str, n: int):
        s = session_or_404(session_id)
        if n >= len(s.watch.events):
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
        rule = body.rule.strip()
        if not rule:
            return JSONResponse(
                {"error": "empty_rule", "hint": "describe something that happens"},
                status_code=400,
            )
        try:
            spec = await perception.normalize(rule)
        except PerceptionError as e:
            return JSONResponse(
                {"error": "perception", "hint": str(e)}, status_code=502
            )
        if not spec.is_transition:
            return JSONResponse(
                {
                    "error": "not_a_transition",
                    "hint": "describe something that happens - e.g. 'the cat jumps onto the table'",
                },
                status_code=400,
            )
        try:
            session = store.create(rule, spec)
        except SessionFull:
            return JSONResponse({"error": "full"}, status_code=503)
        w = session.watch
        return {
            "session_id": session.id,
            "predicate": w.predicate,
            "direction": w.direction,
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
            "events": [{"n": e.n, "at": e.at, "text": e.text} for e in w.events],
        }

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
    return create_app(perception, store, Notifier())
