"""Camera control panel, event cards and Telegram update delivery."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from typing import Optional

import httpx
from loguru import logger

from src.config import MAX_WATCHES
from src.server.cv.perception import Perception, PerceptionError
from src.server.notifier import Notifier
from src.server.session import (
    Event,
    Session,
    SessionStore,
    Subscriber,
    Subscribers,
    Watch,
    add_watch,
    drop_watch,
    new_watch,
    replace_watch,
)
from src.server.telegram.bot import Bot, Buttons

COMMANDS = {
    "menu": "Open your camera dashboard",
    "snapshot": "Take a photo right now",
    "rules": "Add, edit or remove what to watch for",
    "events": "Browse recent moments",
    "pause": "Pause alerts, keep watching",
    "resume": "Turn alerts back on",
    "help": "How it works",
}
SNAPSHOT_TIMEOUT = 8
BACK: Buttons = [[("‹ Dashboard", "menu")]]
WELCOME = (
    "👋 <b>Camera Events</b>\n<i>Be there. From anywhere.</i>\n\n"
    "Your camera watches. You get the moments that matter.\n\n"
    "<b>Get connected</b>\n"
    "① Open the camera page in your browser.\n"
    "② Scan its Telegram QR code.\n"
    "③ Start watching for something that happens.\n\n"
    "Then take a fresh snapshot, manage your rules, and explore events — right here."
)
GONE = (
    "⌛ <b>This link has expired</b>\nReload the camera page and scan its new QR code."
)
NOT_LINKED = (
    "🔗 <b>Connect your camera</b>\nOpen the camera page and scan its Telegram QR code."
)
NO_WATCH = "📷 <b>Your camera is connected</b>\n\nStart a watch on the camera page to unlock snapshots and remote controls."
HELP = (
    "✦ <b>Your camera, within reach</b>\n\n"
    "📸 <b>Snapshot</b> · A new frame, captured after you tap.\n"
    f"✏️ <b>Rules</b> · Up to {MAX_WATCHES} things to watch for at once. Add, edit or remove any.\n"
    "🗂 <b>Recent events</b> · Revisit moments with photo evidence.\n"
    "🔕 <b>Pause alerts</b> · Keep watching without messages. Resume anytime.\n\n"
    "<b>Keep the camera page open</b>\nYour browser supplies the video. If it stops sending frames, snapshots are unavailable.\n\n"
    "Events and connections last for the current server session. Times are in UTC."
)


def safe(text: str, limit: int = 500) -> str:
    return escape(text if len(text) <= limit else text[: limit - 1] + "…")


def menu(sub: Subscriber) -> Buttons:
    return [
        [("📸  Snapshot now", "snapshot")],
        [("✏️ Rules", "rules"), ("🗂 Recent events", "events")],
        [
            (
                ("🔔 Resume alerts", "resume")
                if sub.muted
                else ("🔕 Pause alerts", "pause")
            ),
            ("↻ Refresh", "menu"),
        ],
        [("How it works", "help"), ("Disconnect", "disconnect")],
    ]


@dataclass
class Reply:
    chat_id: int
    text: str
    buttons: Optional[Buttons] = None
    callback_id: Optional[str] = None
    image: Optional[bytes] = None


def _state(w: Watch) -> str:
    if w.tracker.state is None:
        return "⏳ still looking"
    return "✅ True now" if w.tracker.state else "○ False now"


def _watch_lines(s: Session) -> str:
    return "\n".join(
        f"<b>{i + 1:02d}</b> · {safe(w.rule, 250)}\n"
        f"{_state(w)} · <i>{safe(w.evidence, 200) if w.evidence else 'waiting for the first observation…'}</i>"
        for i, w in enumerate(s.watches)
    )


def status_text(s: Session, sub: Subscriber) -> str:
    age = (
        (
            datetime.now(timezone.utc) - datetime.fromisoformat(s.frame_at)
        ).total_seconds()
        if s.frame_at
        else None
    )
    connection = (
        "🟢 Receiving frames"
        if age is not None and age < 15
        else "🟠 Waiting for camera"
    )
    return (
        "📷 <b>CAMERA EVENTS</b>\n<i>Your eyes on what matters.</i>\n\n"
        f"{connection}  ·  {'🔕 Alerts paused' if sub.muted else '🔔 Alerts on'}\n\n"
        f"<b>WATCHING FOR</b>\n{_watch_lines(s)}\n\n"
        f"<b>{len(s.events)}</b> events\n"
        "<i>Take a look, or adjust your rules below.</i>"
    )


def rules_screen(s: Session) -> tuple[str, Buttons]:
    text = (
        f"✏️ <b>YOUR RULES</b>\n<i>{len(s.watches)} of {MAX_WATCHES} · tap to edit or remove</i>\n\n"
        + _watch_lines(s)
    )
    buttons: Buttons = [
        [(f"✏️ {i + 1:02d}", f"edit:{s.id}:{i}"), (f"🗑 {i + 1:02d}", f"drop:{s.id}:{i}")]
        for i in range(len(s.watches))
    ]
    if len(s.watches) < MAX_WATCHES:
        buttons.append([("➕ Add rule", "add")])
    return text, buttons + BACK


def _watch_index(session: Optional[Session], command: str) -> Optional[int]:
    """Parse 'edit:<session>:<i>' / 'drop:<session>:<i>'; None unless it names a live watch."""
    parts = command.split(":")
    if (
        session is None
        or len(parts) != 3
        or parts[1] != session.id
        or not parts[2].isdigit()
        or int(parts[2]) >= len(session.watches)
    ):
        return None
    return int(parts[2])


def caption(w: Watch, event: Event) -> str:
    return (
        "🔔 <b>MOMENT DETECTED</b>\n\n"
        f"<blockquote>{safe(w.rule, 250)}</blockquote>\n"
        f"{safe(w.evidence or event.text, 450)}\n\n"
        f"<i>Event {event.n + 1:02d} · {event.at[:10]} · {event.at[11:19]} UTC</i>"
    )


def watch_of(store: SessionStore, subscriber: Subscriber) -> Optional[Session]:
    return next(
        (s for s in reversed(store.all()) if s.subscriber == subscriber.token), None
    )


def handle(store: SessionStore, subs: Subscribers, update: dict) -> Optional[Reply]:
    callback = update.get("callback_query") or {}
    message = callback.get("message") or update.get("message") or {}
    chat = message.get("chat") or {}
    if "id" not in chat:
        return None
    chat_id = chat["id"]
    # Camera controls belong in the private chat used by the QR link.
    if chat.get("type", "private") != "private":
        return Reply(
            chat_id,
            "🔒 Open a private chat with me to connect and control your camera.",
            callback_id=callback.get("id"),
        )
    text = message.get("text") or ""
    if callback:
        command, arg = callback.get("data", ""), ""
    elif text.startswith("/"):
        command, _, arg = text[1:].partition(" ")
        command = command.split("@")[0].lower()
    else:
        return None
    sub = subs.by_chat(chat_id)
    if command == "start" and arg.strip():
        try:
            linked = subs.get(arg.strip())
        except KeyError:
            return Reply(chat_id, GONE)
        if sub and sub is not linked:
            sub.chat_id = None
            sub.editing = None
        linked.chat_id = chat_id
        linked.editing = None
        sub = linked
    if command in {"help", "start"} and sub is None:
        return Reply(chat_id, WELCOME, [[("How it works", "help_details")]])
    if sub is not None and command != "add" and not command.startswith("edit:"):
        sub.editing = None
    if command in {"help", "help_details"}:
        return Reply(chat_id, HELP, BACK if sub else None, callback.get("id"))
    if sub is None:
        return Reply(chat_id, NOT_LINKED, callback_id=callback.get("id"))
    buttons = menu(sub)
    session = watch_of(store, sub)
    result = Reply(chat_id, "", buttons, callback.get("id"))
    if command in {"menu", "status", "start", "cancel"}:
        result.text = status_text(session, sub) if session else NO_WATCH
    elif command in {"pause", "stop", "resume"}:
        sub.muted = command != "resume"
        result.text = (
            "🔕 <b>Alerts paused</b>\n\nYour camera keeps watching and saving events.\nTap <b>Resume alerts</b> whenever you're ready."
            if sub.muted
            else "🔔 <b>You're back on watch</b>\n\nNew moments will arrive here as they happen."
        )
        result.buttons = menu(sub)
    elif command == "disconnect":
        result.text = "🔗 <b>Disconnect this camera?</b>\n\nYou'll need to scan its QR code again to reconnect. To silence alerts, use Pause instead."
        result.buttons = [
            [("Disconnect camera", "confirm_disconnect")],
            [("Keep connected", "menu")],
        ]
    elif command == "confirm_disconnect":
        sub.chat_id = None
        result.text = "🔗 <b>Camera disconnected</b>\nScan the QR code on your camera page to reconnect."
        result.buttons = None
    elif command in {"rules", "rule"}:
        if session is None:
            result.text = NO_WATCH
        else:
            result.text, result.buttons = rules_screen(session)
    elif command == "add":
        if session is None:
            result.text = NO_WATCH
        elif len(session.watches) >= MAX_WATCHES:
            result.text = f"✏️ <b>Rule limit reached</b>\n\nUp to {MAX_WATCHES} rules per camera. Remove one to add another."
            result.buttons = [[("‹ Rules", "rules")]]
        else:
            sub.editing = "add"
            result.text = (
                "➕ <b>What else should I watch for?</b>\n\n"
                "Send the new rule as a message. Describe a change, for example:\n\n"
                "<i>Someone enters the room</i>\n<i>The cat jumps onto the table</i>\n<i>The door opens</i>\n\n"
                "Your current rules keep running meanwhile."
            )
            result.buttons = [[("Cancel", "rules")]]
    elif command.startswith("edit:"):
        i = _watch_index(session, command)
        if session is None or i is None:
            result.text = "⌛ <b>This rule is no longer there</b>\nOpen Rules to see the current list."
            result.buttons = [[("‹ Rules", "rules")]]
        else:
            sub.editing = f"edit:{i}"
            result.text = (
                f"✏️ <b>Edit rule {i + 1:02d}</b>\n\n"
                f"<b>Now</b>\n<blockquote>{safe(session.watches[i].rule)}</blockquote>\n"
                "Send the new wording as a message. It replaces this rule; its state starts over.\n\n"
                "The current rule keeps running until the new one is ready."
            )
            result.buttons = [[("Cancel", "rules")]]
    elif command.startswith("drop:"):
        i = _watch_index(session, command)
        if session is None or i is None:
            result.text = "⌛ <b>This rule is no longer there</b>\nOpen Rules to see the current list."
            result.buttons = [[("‹ Rules", "rules")]]
        elif len(session.watches) == 1:
            result.text = "✏️ <b>Keep at least one rule</b>\n\nAdd another before removing this one, or stop the watch on the camera page."
            result.buttons = [[("‹ Rules", "rules")]]
        else:
            dropped = session.watches[i]
            drop_watch(session, i)
            result.text, result.buttons = rules_screen(session)
            result.text = (
                f"🗑 <b>Removed</b> · {safe(dropped.rule, 200)}\n\n" + result.text
            )
    elif command == "events":
        if session is None or not session.events:
            result.text = "🗂 <b>The best moments go here</b>\n\nNo events yet. When your rule triggers, you'll find the photo and time here."
            result.buttons = BACK
        else:
            recent = list(reversed(session.events[-5:]))
            result.text = (
                "🗂 <b>RECENT MOMENTS</b>\n<i>Latest five events · tap to see the photo</i>\n\n"
                + "\n\n".join(
                    f"<b>{e.n + 1:02d}</b> · {e.at[11:19]} UTC\n{safe(e.rule or e.text, 180)}"
                    for e in recent
                )
            )
            result.buttons = [
                [
                    (
                        f"📷 Event {e.n + 1:02d} · {e.at[11:19]}",
                        f"event:{session.id}:{e.n}",
                    )
                ]
                for e in recent
            ] + BACK
    elif command.startswith("event:"):
        parts = command.split(":")
        if (
            session is None
            or len(parts) != 3
            or parts[1] != session.id
            or not parts[2].isdigit()
            or int(parts[2]) >= len(session.events)
        ):
            result.text = "⌛ <b>This moment is no longer available</b>\nOpen Recent events to see this watch's moments."
        else:
            event = session.events[int(parts[2])]
            result.image = event.image
            result.text = (
                f"🗂 <b>MOMENT {event.n + 1:02d}</b>\n\n"
                + (
                    f"<blockquote>{safe(event.rule, 250)}</blockquote>\n"
                    if event.rule
                    else ""
                )
                + f"{safe(event.text, 650)}\n\n<i>{event.at[:10]} · {event.at[11:19]} UTC</i>"
            )
            result.buttons = [[("‹ Recent events", "events"), ("Dashboard", "menu")]]
    elif command == "snapshot":
        result.text = (
            NO_WATCH if session is None else "📸 <b>Waiting for a fresh frame…</b>"
        )
    else:
        result.text = "✦ <b>Your camera controls are below</b>\nTo change what I watch for, open <b>Rules</b> first."
    return result


async def respond(
    bot: Bot,
    store: SessionStore,
    subs: Subscribers,
    perception: Perception,
    update: dict,
) -> None:
    callback = update.get("callback_query") or {}
    message = callback.get("message") or update.get("message") or {}
    chat = message.get("chat") or {}
    if callback:
        await bot.answer_callback(callback["id"])
    if "id" not in chat:
        return
    chat_id = chat["id"]
    sub = subs.by_chat(chat_id)
    text = (message.get("text") or "").strip()
    command = (
        callback.get("data", "")
        if callback
        else (
            text.split(" ")[0].split("@")[0][1:].lower() if text.startswith("/") else ""
        )
    )
    reply = handle(store, subs, update)
    private = chat.get("type", "private") == "private"
    if (
        private
        and sub
        and not callback
        and (
            sub.editing
            and not text.startswith("/")
            or command in {"rule", "rules"}
            and " " in text
        )
    ):
        mode = sub.editing or "add"  # "/rules <text>" adds
        rule = text.partition(" ")[2].strip() if text.startswith("/") else text
        session = watch_of(store, sub)
        if session is None:
            reply = Reply(chat_id, NO_WATCH, BACK)
        elif not rule or len(rule) > 500:
            reply = Reply(
                chat_id,
                "✏️ Please send a rule between 1 and 500 characters.",
                [[("Cancel", "cancel")]],
            )
        else:
            await bot.chat_action(chat_id, "typing")
            try:
                spec = await perception.normalize(rule)
            except PerceptionError:
                reply = Reply(
                    chat_id,
                    "⚠️ <b>Couldn't understand that rule</b>\nYour current rules are still running. Please try again.",
                    [[("Cancel", "rules")]],
                )
            else:
                session.usage += spec.usage
                if (
                    session.closed
                    or sub.chat_id != chat_id
                    or watch_of(store, sub) is not session
                ):
                    reply = Reply(
                        chat_id,
                        "⌛ Your camera session changed. Open the dashboard and try again.",
                        BACK,
                    )
                elif not spec.is_transition:
                    reply = Reply(
                        chat_id,
                        "✏️ <b>Describe something that changes</b>\nTry: <i>Someone enters the room</i>.\nYour rules are unchanged.",
                        [[("Cancel", "rules")]],
                    )
                else:
                    watch = new_watch(rule, spec)
                    done = (
                        add_watch(session, watch, MAX_WATCHES)
                        if mode == "add"
                        else replace_watch(session, int(mode.partition(":")[2]), watch)
                    )
                    sub.editing = None
                    if not done:
                        reply = Reply(
                            chat_id,
                            "⌛ <b>Your rules changed meanwhile</b>\nOpen Rules and try again.",
                            [[("‹ Rules", "rules")]],
                        )
                    else:
                        reply = Reply(
                            chat_id,
                            (
                                "➕ <b>Rule added</b>"
                                if mode == "add"
                                else "✅ <b>Rule updated</b>"
                            )
                            + "\n\n"
                            + status_text(session, sub),
                            menu(sub),
                        )
    elif private and command == "snapshot" and sub:
        session = watch_of(store, sub)
        if session:
            await bot.chat_action(chat_id, "upload_photo")
            session.frame_received.clear()
            try:
                await asyncio.wait_for(
                    session.frame_received.wait(), timeout=SNAPSHOT_TIMEOUT
                )
            except asyncio.TimeoutError:
                reply = Reply(
                    chat_id,
                    "🟠 <b>Camera isn't sending frames</b>\n\nKeep the camera page open with a watch running, then try again.",
                    [[("📸 Try again", "snapshot")]] + BACK,
                )
            else:
                if (
                    session.closed
                    or sub.chat_id != chat_id
                    or watch_of(store, sub) is not session
                ):
                    reply = Reply(
                        chat_id,
                        "⌛ Your camera session changed. Please try again.",
                        BACK,
                    )
                else:
                    reply = Reply(
                        chat_id,
                        f"📸 <b>RIGHT NOW</b>\n\nFresh from your camera.\n<i>{session.frame_at[:10]} · {session.frame_at[11:19]} UTC</i>",
                        [[("📸 Take another", "snapshot"), ("Dashboard", "menu")]],
                        image=session.latest_frame,
                    )
    if reply is None:
        return
    if reply.image is not None:
        await bot.send_photo(chat_id, reply.image, reply.text, reply.buttons)
    elif callback and "text" in message and message.get("message_id"):
        await bot.edit_message(
            chat_id, message["message_id"], reply.text, reply.buttons
        )
    else:
        await bot.send_message(chat_id, reply.text, reply.buttons)


class TelegramNotifier(Notifier):
    def __init__(self, bot: Bot, store: SessionStore, subs: Subscribers) -> None:
        self.bot, self.store, self.subs = bot, store, subs

    async def notify(self, session_id: str, watch: Watch, event: Event) -> None:
        await super().notify(session_id, watch, event)
        try:
            token = self.store.get(session_id).subscriber
            sub = self.subs.get(token) if token else None
        except KeyError:
            return
        if sub is None or sub.chat_id is None or sub.muted:
            return
        try:
            await self.bot.send_photo(
                sub.chat_id,
                event.image,
                caption(watch, event),
                [
                    [("📸 See now", "snapshot"), ("🗂 Recent events", "events")],
                    [("🔕 Pause alerts", "pause"), ("Dashboard", "menu")],
                ],
            )
        except httpx.HTTPError as e:
            logger.warning("session={} telegram send failed: {}", session_id, e)


async def poll(
    bot: Bot, store: SessionStore, subs: Subscribers, perception: Perception
) -> None:
    """One polling instance. Updates are ordered so rule edits and cancellation agree."""
    await bot.set_commands(COMMANDS)
    offset = 0
    while True:
        try:
            updates = await bot.get_updates(offset)
        except httpx.HTTPError as e:
            logger.warning("telegram poll failed: {!r}", e)
            await asyncio.sleep(3)
            continue
        for update in updates:
            offset = update["update_id"] + 1
            try:
                await respond(bot, store, subs, perception, update)
            except httpx.HTTPError as e:
                logger.warning("telegram reply failed: {}", e)
