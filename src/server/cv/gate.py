"""Which frames are worth a model call. Port of notebooks/camera_v3.ipynb, cells 6 and 10."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

GATE_SIZE = (128, 72)  # width, height; 16x9 px per tile
GRID = 8  # 8x8 board of tiles
THRESHOLD = 6.6  # Lab levels in the loudest tile; recalibrate on your empty room
GLOBAL = (
    0.5  # loud tiles over this share of the board: light or exposure, not an object
)
PERSIST = 2  # frames in a row a change must hold before it is worth sending
OVERLAP = 0.5  # share of loud tiles that must repeat from the previous frame


def decode(image: bytes) -> np.ndarray:
    """JPEG/PNG bytes -> RGB uint8 array."""
    bgr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("not a decodable image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def prep(x: np.ndarray) -> np.ndarray:
    """Small, blurred Lab. L is centred: a lamp moves the mean, not the room."""
    x = cv2.resize(x, GATE_SIZE, interpolation=cv2.INTER_AREA)
    x = cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_RGB2LAB), (0, 0), 2)
    x = x.astype(np.float32)
    x[..., 0] -= x[..., 0].mean()
    return x


def board(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """GRID x GRID board: mean of the loudest channel difference per tile."""
    diff = np.abs(prep(a) - prep(b)).max(axis=2)
    h, w = diff.shape
    return diff.reshape(GRID, h // GRID, GRID, w // GRID).mean(axis=(1, 3))


def verdict(mask: np.ndarray) -> str:
    """mask = board > THRESHOLD. 'skip' | 'change' | 'light'; only 'change' can lead to a call."""
    # ponytail: a person filling the frame also reads as 'light'; add stability check if that bites
    return "light" if mask.mean() > GLOBAL else "change" if mask.any() else "skip"


@dataclass
class GateResult:
    verdict: str  # first | skip | change | light
    streak: int
    send: bool


class Gate:
    """Per-session gate state: the anchor and how long the current change has held."""

    def __init__(self) -> None:
        self.anchor: Optional[np.ndarray] = None
        self.streak = 0
        self.prev: Optional[np.ndarray] = None

    def observe(self, frame: np.ndarray) -> GateResult:
        if self.anchor is None:
            self.anchor = frame
            return GateResult("first", 0, True)
        mask = board(self.anchor, frame) > THRESHOLD
        v = verdict(mask)
        if v == "skip":
            self.streak, self.prev = 0, None
            return GateResult(v, 0, False)
        same = self.prev is not None and (mask & self.prev).sum() / mask.sum() > OVERLAP
        self.streak = self.streak + 1 if same else 1
        self.prev = mask
        result = GateResult(v, self.streak, self.streak >= PERSIST and v == "change")
        if self.streak >= PERSIST:  # change confirmed or light settled: new baseline
            self.anchor, self.streak, self.prev = frame, 0, None
        return result
