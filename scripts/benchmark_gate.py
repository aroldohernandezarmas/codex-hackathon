"""Synthetic nuisance/object checks on local stills; no paid API calls.

Run: poetry run python -m scripts.benchmark_gate
This measures pair decisions, not real-world event recall or monthly savings.
"""

import csv
from time import perf_counter

import cv2
import numpy as np
from loguru import logger

from src.config import DATA_DIR
from src.server.cv.gate import THRESHOLD, board, decode


def legacy_send(a: np.ndarray, b: np.ndarray) -> bool:
    """Frozen pre-change algorithm, including its >50% 'light' rejection."""

    def prep(frame):
        frame = cv2.resize(frame, (128, 72), interpolation=cv2.INTER_AREA)
        frame = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_RGB2LAB), (0, 0), 2)
        frame = frame.astype(np.float32)
        frame[..., 0] -= frame[..., 0].mean()
        return frame

    diff = np.abs(prep(a) - prep(b)).max(axis=2)
    mask = diff.reshape(8, 9, 8, 16).mean(axis=(1, 3)) > THRESHOLD
    return bool(mask.any() and mask.mean() <= 0.5)


def cases(base):
    yield "identical", base, base, False
    for dx in (2, 4, 8, 12):
        shifted = cv2.warpAffine(
            base,
            np.float32([[1, 0, dx], [0, 1, -dx / 2]]),
            (1280, 720),
            borderMode=cv2.BORDER_REFLECT_101,
        )
        yield f"camera_{dx}px", base, shifted, False
    for name, gain, offset in (
        ("brighter", 1, 20),
        ("contrast", 1.2, -10),
        ("darker", 0.8, 10),
        ("warm", 1, [15, 0, -10]),
    ):
        lit = np.clip(base.astype(float) * gain + offset, 0, 255).astype(np.uint8)
        yield name, base, lit, False
    obj = base.copy()
    obj[270:450, 480:640] = [200, 40, 40]
    yield "object", base, obj, True
    lit = np.clip(obj.astype(float) * 1.2 + 10, 0, 255).astype(np.uint8)
    moved = cv2.warpAffine(
        lit,
        np.float32([[1, 0, 8], [0, 1, -4]]),
        (1280, 720),
        borderMode=cv2.BORDER_REFLECT_101,
    )
    yield "object_light_camera", base, moved, True
    large = base.copy()
    large[50:650, 160:1120] = [30, 80, 180]
    yield "large_object", base, large, True
    before, after = base.copy(), base.copy()
    before[270:450, 480:640] = 0
    after[270:450, 496:656] = 0
    yield "object_moves_16px", before, after, True


def main():
    rows = []
    for path in sorted(DATA_DIR.glob("*.png")):
        base = cv2.resize(decode(path.read_bytes()), (1280, 720))
        base = (30 + base.astype(float) * 0.7).astype(np.uint8)
        board(base, np.clip(base.astype(float) + 20, 0, 255).astype(np.uint8))
        for name, before, after, expected in cases(base):
            start = perf_counter()
            score = float(board(before, after).max())
            elapsed = (perf_counter() - start) * 1000
            rows.append(
                dict(
                    image=path.name,
                    case=name,
                    expected_send=expected,
                    old_send=legacy_send(before, after),
                    new_send=score > THRESHOLD,
                    score=round(score, 3),
                    milliseconds=round(elapsed, 3),
                )
            )
    if not rows:
        raise SystemExit(f"No PNG samples in {DATA_DIR}")
    output = DATA_DIR / "gate-benchmark.csv"
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for key in ("old_send", "new_send"):
        false_calls = sum(row[key] and not row["expected_send"] for row in rows)
        missed = sum(not row[key] and row["expected_send"] for row in rows)
        logger.info("{}: false calls={}, missed changes={}", key, false_calls, missed)
    logger.info(
        "{} pairs; median={:.1f}ms, p95={:.1f}ms; {}",
        len(rows),
        np.median([row["milliseconds"] for row in rows]),
        np.percentile([row["milliseconds"] for row in rows], 95),
        output,
    )


if __name__ == "__main__":
    main()
