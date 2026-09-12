import importlib

from src import config


def test_defaults(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.setenv("DATA_DIR", "/tmp/skeleton-data")
    reloaded = importlib.reload(config)
    assert reloaded.LOG_LEVEL == "INFO"
    assert str(reloaded.DATA_DIR) == "/tmp/skeleton-data"
