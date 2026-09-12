"""Точка входа: uvicorn поверх src.server.app."""

import sys

import uvicorn
from loguru import logger

from src import config
from src.server.app import default_app


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level=config.LOG_LEVEL)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("started, data_dir={} port={}", config.DATA_DIR, config.PORT)
    uvicorn.run(default_app(), host="0.0.0.0", port=config.PORT, log_level="warning")


if __name__ == "__main__":
    main()
