"""Thin Telegram Bot API client. Only the calls the notifier and the binder need."""

import io
import json
from typing import Any, Optional

import httpx
import segno

BASE_URL = "https://api.telegram.org"
POLL_SECONDS = 25  # long-poll wait; the client timeout stays above it

Buttons = list[list[tuple[str, str]]]  # rows of (label, callback_data)


def _markup(buttons: Optional[Buttons]) -> Optional[dict]:
    if not buttons:
        return None
    rows = [[{"text": t, "callback_data": d} for t, d in row] for row in buttons]
    return {"inline_keyboard": rows}


class Bot:
    def __init__(self, token: str) -> None:
        if not token:
            raise ValueError("no TELEGRAM_BOT_TOKEN configured")
        self.client = httpx.AsyncClient(
            base_url=f"{BASE_URL}/bot{token}", timeout=POLL_SECONDS + 10
        )
        self.username: Optional[str] = None  # filled by get_me()

    async def aclose(self) -> None:
        await self.client.aclose()

    async def _call(self, method: str, **kwargs: Any) -> Any:
        response = await self.client.post(f"/{method}", **kwargs)
        response.raise_for_status()
        return response.json()["result"]

    async def get_me(self) -> str:
        self.username = str((await self._call("getMe"))["username"])
        return self.username

    async def set_commands(self, commands: dict[str, str]) -> None:
        """The "/" menu in the chat: {"status": "what I see now", ...}."""
        payload = [{"command": c, "description": d} for c, d in commands.items()]
        await self._call("setMyCommands", json={"commands": payload})

    def deep_link(self, code: str) -> str:
        """t.me link that sends the bot `/start <code>`."""
        return f"https://t.me/{self.username}?start={code}"

    def qr_svg(self, code: str) -> bytes:
        """The deep link as an SVG QR code: scan with the phone camera to bind the chat."""
        buffer = io.BytesIO()
        segno.make(self.deep_link(code), error="m").save(
            buffer, kind="svg", scale=8, border=2, dark="#000", light="#fff"
        )
        return buffer.getvalue()

    async def get_updates(self, offset: int) -> list[dict]:
        payload = {
            "offset": offset,
            "timeout": POLL_SECONDS,
            "allowed_updates": ["message", "callback_query"],
        }
        return list(await self._call("getUpdates", json=payload))

    async def send_message(
        self, chat_id: int, text: str, buttons: Optional[Buttons] = None
    ) -> None:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
        }
        if buttons:
            payload["reply_markup"] = _markup(buttons)
        await self._call("sendMessage", json=payload)

    async def send_photo(
        self, chat_id: int, jpeg: bytes, caption: str, buttons: Optional[Buttons] = None
    ) -> None:
        data = {"chat_id": str(chat_id), "caption": caption, "parse_mode": "HTML"}
        if buttons:
            data["reply_markup"] = json.dumps(_markup(buttons))
        await self._call(
            "sendPhoto", data=data, files={"photo": ("event.jpg", jpeg, "image/jpeg")}
        )

    async def answer_callback(self, callback_id: str) -> None:
        """Stops the spinner on the pressed button."""
        await self._call("answerCallbackQuery", json={"callback_query_id": callback_id})
