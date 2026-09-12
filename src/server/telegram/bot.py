"""Thin Telegram Bot API client. Only the four calls the notifier and the binder need."""

from typing import Any, Optional

import httpx

BASE_URL = "https://api.telegram.org"
POLL_SECONDS = 25  # long-poll wait; the client timeout stays above it


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

    def deep_link(self, code: str) -> str:
        """t.me link that sends the bot `/start <code>`; the page draws it as a QR."""
        return f"https://t.me/{self.username}?start={code}"

    async def get_updates(self, offset: int) -> list[dict]:
        payload = {
            "offset": offset,
            "timeout": POLL_SECONDS,
            "allowed_updates": ["message"],
        }
        return list(await self._call("getUpdates", json=payload))

    async def send_message(self, chat_id: int, text: str) -> None:
        await self._call("sendMessage", json={"chat_id": chat_id, "text": text})

    async def send_photo(self, chat_id: int, jpeg: bytes, caption: str) -> None:
        await self._call(
            "sendPhoto",
            data={"chat_id": str(chat_id), "caption": caption},
            files={"photo": ("event.jpg", jpeg, "image/jpeg")},
        )
