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
XAI_REQUEST_TIMEOUT = float(os.getenv("XAI_REQUEST_TIMEOUT", "10"))
XAI_ATTEMPT_TIMEOUT = float(os.getenv("XAI_ATTEMPT_TIMEOUT", "5"))
# USD per 1M tokens, used only to estimate spend in the UI and logs
XAI_PRICE_PROMPT = float(os.getenv("XAI_PRICE_PROMPT", "3.0"))
XAI_PRICE_COMPLETION = float(os.getenv("XAI_PRICE_COMPLETION", "15.0"))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "10"))
SESSION_TTL = float(os.getenv("SESSION_TTL", "30"))
MAX_SUBSCRIBERS = int(os.getenv("MAX_SUBSCRIBERS", "20"))
SUBSCRIBER_TTL = float(os.getenv("SUBSCRIBER_TTL", "86400"))
PORT = int(os.getenv("PORT", "8000"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
