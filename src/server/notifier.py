"""Where events go. Logs only until the Telegram phase."""

from loguru import logger

from src.server.session import Event, Watch


class Notifier:
    async def notify(self, session_id: str, watch: Watch, event: Event) -> None:
        # ponytail: Telegram lands here — photo + caption to the chat bound to session_id
        logger.info(
            "EVENT session={} n={} {} ({} bytes)",
            session_id,
            event.n,
            event.text,
            len(event.image),
        )
