from pathlib import Path

import cv2
import numpy as np
import pytest

from src.server.cv.gate import PERSIST, THRESHOLD, Gate, board, decode

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def base() -> np.ndarray:
    rgb = decode((DATA / "1.png").read_bytes())
    rgb = cv2.resize(rgb, (1280, 720), interpolation=cv2.INTER_AREA)
    return (30 + rgb.astype(np.float32) * 0.7).astype(np.uint8)  # headroom for +20


def test_lamp_is_quiet_and_object_is_loud(base):
    lamp = np.clip(base.astype(np.int16) + 20, 0, 255).astype(np.uint8)
    covered = base.copy()
    covered[:90, :160] = 0
    assert board(base, base).max() == 0
    assert board(base, lamp).max() < THRESHOLD < board(base, covered).max()


def test_gate_sends_first_then_needs_persist(base):
    covered = base.copy()
    covered[:90, :160] = 0
    gate = Gate()
    assert gate.observe(base).send is True  # first frame
    assert gate.observe(base).send is False  # same picture: skip
    assert gate.observe(covered).streak == 1  # change, not persisted yet
    two = gate.observe(covered)
    assert (two.streak, two.send) == (PERSIST, True)
    assert gate.observe(covered).verdict == "skip"  # anchor moved
