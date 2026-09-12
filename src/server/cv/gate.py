"""Compare scene content after compensating small camera and lighting changes."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

GATE_SIZE = (128, 72)  # width, height; 16x9 px per tile
GRID = 8  # 8x8 board of tiles
THRESHOLD = 6.6  # Lab levels in the loudest tile; recalibrate on your empty room


def decode(image: bytes) -> np.ndarray:
    """JPEG/PNG bytes -> RGB uint8 array."""
    if not image:
        raise ValueError("not a decodable image")
    bgr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("not a decodable image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def prep(x: np.ndarray) -> np.ndarray:
    """Small, blurred RGB; compensate exposure before converting to Lab."""
    x = cv2.resize(x, GATE_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(x.astype(np.float32), (0, 0), 2)


def align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Accept only a small translation supported across the textured background."""
    valid = np.ones(a.shape[:2], np.uint8)
    gray_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    try:
        correlation, warp = cv2.findTransformECC(
            gray_a,
            gray_b,
            np.eye(2, 3, dtype=np.float32),
            cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.001),
        )
    except cv2.error:  # flat images or unrelated scenes cannot be registered
        return b, valid
    if correlation < 0.97 or np.abs(warp[:, 2]).max() > 2:
        return b, valid
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    moved = cv2.warpAffine(b, warp, GATE_SIZE, flags=flags)
    overlap = cv2.warpAffine(valid.astype(np.float32), warp, GATE_SIZE, flags=flags)
    overlap = (overlap > 0.999).astype(np.uint8)
    gray_moved = cv2.cvtColor(moved, cv2.COLOR_RGB2GRAY)
    h, w = gray_a.shape
    supported = 0
    quadrants = set()
    for y in range(0, h, h // 4):
        for x in range(0, w, w // 4):
            region = np.s_[y : y + h // 4, x : x + w // 4]
            mask = overlap[region].astype(bool)
            left, right = gray_a[region][mask], gray_moved[region][mask]
            if left.std() > 3 and right.std() > 3:
                before = gray_b[region][mask]
                if before.std() > 3:
                    original = np.corrcoef(left, before)[0, 1]
                    corrected = np.corrcoef(left, right)[0, 1]
                    if corrected > max(0.95, original + 0.0001):
                        supported += 1
                        quadrants.add((y // (h // 2), x // (w // 2)))
    return (moved, overlap) if supported >= 8 and len(quadrants) >= 3 else (b, valid)


def compensate_light(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fit RGB gain/offset to the quiet majority, retaining local object residuals."""
    x, y = b[valid.astype(bool)], a[valid.astype(bool)]
    gain = np.ones(3, np.float32)
    offset = np.median(y - x, axis=0)
    for _ in range(3):
        error = np.abs(y - (x * gain + offset)).max(axis=1)
        keep = error <= np.quantile(error, 0.7)
        xx, yy = x[keep], y[keep]
        xc, yc = xx - xx.mean(axis=0), yy - yy.mean(axis=0)
        variance = (xc * xc).mean(axis=0)
        gain = np.divide(
            (xc * yc).mean(axis=0),
            variance,
            out=np.ones(3, np.float32),
            where=variance > 4,
        )
        offset = yy.mean(axis=0) - gain * xx.mean(axis=0)
    if np.any((gain < 0.67) | (gain > 1.5)) or np.abs(offset).max() > 40:
        return b
    corrected = np.clip(b * gain + offset, 0, 255)
    quiet = (np.abs(a - corrected).max(axis=2) < 8) & valid.astype(bool)
    h, w = valid.shape
    # A local foreground must not define the exposure of the entire image.
    for y0 in (0, h // 2):
        for x0 in (0, w // 2):
            region = np.s_[y0 : y0 + h // 2, x0 : x0 + w // 2]
            if quiet[region].sum() < valid[region].sum() * 0.5:
                return b
    return corrected


def board(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """GRID x GRID residual Lab difference over the actual overlapping pixels."""
    a, b = prep(a), prep(b)
    if np.array_equal(a, b):
        return np.zeros((GRID, GRID), np.float32)
    b, valid = align(a, b)
    b = compensate_light(a, b, valid)
    a = cv2.cvtColor(np.rint(a).astype(np.uint8), cv2.COLOR_RGB2LAB)
    b = cv2.cvtColor(np.rint(b).astype(np.uint8), cv2.COLOR_RGB2LAB)
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).max(axis=2)
    h, w = diff.shape
    shape = (GRID, h // GRID, GRID, w // GRID)
    total = (diff * valid).reshape(shape).sum(axis=(1, 3))
    count = valid.reshape(shape).sum(axis=(1, 3))
    return total / np.maximum(count, 1)


def verdict(mask: np.ndarray) -> str:
    """Large residuals are real candidates too; area alone cannot identify light."""
    return "change" if mask.any() else "skip"


@dataclass
class GateResult:
    verdict: str  # first | skip | change
    streak: int
    send: bool


class Gate:
    """Per-session anchor; send the first changed frame without waiting."""

    def __init__(self) -> None:
        self.anchor: Optional[np.ndarray] = None

    def observe(self, frame: np.ndarray, advance: bool = True) -> GateResult:
        if self.anchor is None:
            if advance:
                self.anchor = frame
            return GateResult("first", 0, True)
        mask = board(self.anchor, frame) > THRESHOLD
        v = verdict(mask)
        if v == "skip":
            return GateResult(v, 0, False)
        if advance:  # only consume changes we can process
            self.anchor = frame
        return GateResult(v, 1, v == "change")
