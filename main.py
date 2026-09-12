"""Точка входа."""

import sys

from loguru import logger

from src import config


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level=config.LOG_LEVEL)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("started, data_dir={}", config.DATA_DIR)


if __name__ == "__main__":
    main()
