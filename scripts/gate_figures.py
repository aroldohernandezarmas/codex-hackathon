"""Render README figures: per-case triptych (anchor | frame | 8x8 heatmap) via the real gate."""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.server.cv.gate import (
    GRID,
    THRESHOLD,
    align,
    board,
    compensate_light,
    decode,
    prep,
)

OUT = "docs/images"
W, H = 480, 270


def load(p):
    return cv2.resize(decode(open(p, "rb").read()), (1280, 720))


def heat(b, hi=20.0):
    n = np.clip(b / hi, 0, 1)
    img = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    th, tw = H // GRID, W // GRID
    for y in range(GRID):
        for x in range(GRID):
            v = b[y, x]
            if v > THRESHOLD:
                d.rectangle(
                    [x * tw + 1, y * th + 1, (x + 1) * tw - 2, (y + 1) * th - 2],
                    outline=(255, 255, 255),
                    width=2,
                )
    return pil


def font(sz):
    for f in ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf"]:
        try:
            return ImageFont.truetype(f, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def panel(rgb, label):
    pil = Image.fromarray(
        cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    ).convert("RGB")
    d = ImageDraw.Draw(pil)
    d.rectangle([0, 0, W, 28], fill=(0, 0, 0))
    d.text((8, 5), label, fill=(255, 255, 255), font=font(16))
    return pil


def triptych(name, a, b, caption):
    bd = board(a, b)
    peak = float(bd.max())
    verdict = "SEND to model" if peak > THRESHOLD else "SKIP (free)"
    tiles = int((bd > THRESHOLD).sum())
    hm = heat(bd)
    d = ImageDraw.Draw(hm)
    d.rectangle([0, 0, W, 28], fill=(0, 0, 0))
    d.text(
        (8, 5),
        f"residual heatmap  peak {peak:.1f}  threshold {THRESHOLD}  -> {verdict}",
        fill=(255, 255, 255),
        font=font(15),
    )
    canvas = Image.new("RGB", (W * 3 + 16, H + 40), (14, 17, 22))
    canvas.paste(panel(a, "anchor (last frame sent)"), (0, 0))
    canvas.paste(panel(b, caption), (W + 8, 0))
    canvas.paste(hm, (2 * W + 16, 0))
    ImageDraw.Draw(canvas).text(
        (8, H + 10),
        f"{name}: {tiles} of {GRID*GRID} tiles above threshold",
        fill=(220, 220, 220),
        font=font(16),
    )
    canvas.save(f"{OUT}/gate-{name}.png", optimize=True)
    print(name, f"peak={peak:.2f}", verdict, tiles)


base, cat = load("data/1.png"), load("data/2.png")
triptych("cat", base, cat, "5 s later: the cat arrives")

shift = cv2.warpAffine(
    base,
    np.float32([[1, 0, 9], [0, 1, -5]]),
    (1280, 720),
    borderMode=cv2.BORDER_REFLECT_101,
)
triptych("camera-nudge", base, shift, "phone nudged ~1% of frame")

lit = np.clip(base.astype(float) * 0.8 - 6, 0, 255).astype(np.uint8)
triptych("lighting", base, lit, "exposure -20%, lamp dimmed")

both = cv2.warpAffine(
    np.clip(cat.astype(float) * 0.8 - 6, 0, 255).astype(np.uint8),
    np.float32([[1, 0, 9], [0, 1, -5]]),
    (1280, 720),
    borderMode=cv2.BORDER_REFLECT_101,
)
triptych("cat-nudge-light", base, both, "cat + nudge + dimmed")


# naive pixel diff vs gate pipeline, for the lighting case: show what compensation removes
def naive(a, b):
    a, b = prep(a), prep(b)
    a = cv2.cvtColor(np.rint(a).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(np.rint(b).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    diff = np.abs(a - b).max(axis=2)
    h, w = diff.shape
    return diff.reshape(GRID, h // GRID, GRID, w // GRID).mean(axis=(1, 3))


for nm, fr in (("lighting", lit), ("camera-nudge", shift)):
    nv, gt = naive(base, fr), board(base, fr)
    row = Image.new("RGB", (W * 2 + 8, H + 40), (14, 17, 22))
    l = heat(nv)
    ImageDraw.Draw(l).rectangle([0, 0, W, 28], fill=(0, 0, 0))
    ImageDraw.Draw(l).text(
        (8, 5), f"raw Lab diff  peak {nv.max():.1f}", fill="white", font=font(15)
    )
    r = heat(gt)
    ImageDraw.Draw(r).rectangle([0, 0, W, 28], fill=(0, 0, 0))
    ImageDraw.Draw(r).text(
        (8, 5),
        f"after align + light fit  peak {gt.max():.1f}",
        fill="white",
        font=font(15),
    )
    row.paste(l, (0, 0))
    row.paste(r, (W + 8, 0))
    ImageDraw.Draw(row).text(
        (8, H + 10),
        f"{nm}: same pair, before and after compensation",
        fill=(220, 220, 220),
        font=font(16),
    )
    row.save(f"{OUT}/gate-{nm}-before-after.png", optimize=True)
    print(nm, "naive", f"{nv.max():.1f}", "gate", f"{gt.max():.1f}")
