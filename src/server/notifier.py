"""Where events go. Logs only until the Telegram phase."""

from loguru import logger

from src.server.session import Event, Session


class Notifier:
    async def notify(self, session: Session, event: Event) -> None:
        # ponytail: Telegram lands here — photo + caption to the chat bound to session.id
        logger.info(
            "EVENT session={} n={} {} ({} bytes)",
            session.id,
            event.n,
            event.text,
            len(event.image),
        )
