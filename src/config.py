"""Настройки из переменных окружения."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
XAI_API_KEYS = [
    k.strip() for k in os.getenv("XAI_API_KEYS", "").split(",") if k.strip()
]
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4.20-0309-non-reasoning")
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "10"))
SESSION_TTL = float(os.getenv("SESSION_TTL", "30"))
PORT = int(os.getenv("PORT", "8000"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
