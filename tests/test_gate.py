from pathlib import Path

import cv2
import numpy as np
import pytest

from src.server.cv.gate import THRESHOLD, Gate, board, decode

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(params=["1.png", "2.png", "3.png"])
def base(request) -> np.ndarray:
    rgb = decode((DATA / request.param).read_bytes())
    rgb = cv2.resize(rgb, (1280, 720), interpolation=cv2.INTER_AREA)
    return (30 + rgb.astype(np.float32) * 0.7).astype(np.uint8)  # headroom for +20


def test_lamp_is_quiet_and_object_is_loud(base):
    lamp = np.clip(base.astype(np.int16) + 20, 0, 255).astype(np.uint8)
    covered = base.copy()
    covered[:90, :160] = 0
    assert board(base, base).max() == 0
    assert board(base, lamp).max() < THRESHOLD < board(base, covered).max()


def test_gate_sends_first_changed_frame_immediately(base):
    covered = base.copy()
    covered[:90, :160] = 0
    gate = Gate()
    assert gate.observe(base).send is True  # first frame
    assert gate.observe(base).send is False  # same picture: skip
    changed = gate.observe(covered)
    assert (changed.verdict, changed.send) == ("change", True)
    assert gate.observe(covered).verdict == "skip"  # anchor moved


def transform(frame, dx=0, dy=0, gain=1, offset=0):
    lit = np.clip(frame.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)
    return cv2.warpAffine(
        lit,
        np.float32([[1, 0, dx], [0, 1, dy]]),
        (frame.shape[1], frame.shape[0]),
        borderMode=cv2.BORDER_REFLECT_101,
    )


@pytest.mark.parametrize(
    "params",
    [
        {"dx": 2, "dy": -1},
        {"dx": -4, "dy": 2},
        {"dx": 8, "dy": -4},
        {"dx": -12, "dy": 6},
        {"gain": 1.2, "offset": -10},
        {"gain": 0.8, "offset": 10},
        {"offset": [15, 0, -10]},
        {"dx": 8, "dy": -4, "gain": 1.2, "offset": 10},
    ],
)
def test_camera_and_exposure_changes_do_not_call_model(base, params):
    gate = Gate()
    gate.observe(base)
    assert not gate.observe(transform(base, **params)).send
    assert gate.anchor is base  # nuisance frames must not consume unseen changes


@pytest.mark.parametrize("dx,gain", [(0, 1), (8, 1), (0, 1.2), (8, 1.2)])
def test_object_is_detected_immediately_despite_camera_or_light_change(base, dx, gain):
    changed = base.copy()
    changed[270:450, 480:640] = [200, 40, 40]
    gate = Gate()
    gate.observe(base)
    assert gate.observe(transform(changed, dx=dx, dy=-dx / 2, gain=gain)).send


def test_large_object_is_not_mistaken_for_light(base):
    changed = base.copy()
    changed[50:650, 160:1120] = [30, 80, 180]
    gate = Gate()
    gate.observe(base)
    assert gate.observe(changed).send


def test_small_object_motion_accumulates_against_sent_anchor(base):
    def scene(dx):
        frame = base.copy()
        frame[270:450, 480 + dx : 640 + dx] = 0
        return frame

    gate = Gate()
    gate.observe(scene(0))
    assert any(gate.observe(scene(dx)).send for dx in (2, 4, 8, 16))


def test_busy_model_does_not_consume_change(base):
    gate = Gate()
    gate.observe(base)
    changed = base.copy()
    changed[:90, :160] = 0
    assert gate.observe(changed, advance=False).send
    assert gate.anchor is base
    assert gate.observe(changed).send


def test_registration_failure_still_detects_objects(base, monkeypatch):
    def fail(*args):
        raise cv2.error("cannot align")

    monkeypatch.setattr(cv2, "findTransformECC", fail)
    changed = base.copy()
    changed[:90, :160] = 0
    gate = Gate()
    gate.observe(base)
    assert gate.observe(changed).send


def test_flat_background_does_not_hide_foreground():
    base = np.full((720, 1280, 3), 150, np.uint8)
    changed = base.copy()
    changed[:90, :160] = 0
    gate = Gate()
    gate.observe(base)
    assert gate.observe(changed).send
