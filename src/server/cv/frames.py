"""Frame source. Real frames arrive as bytes from the UI (browser/phone camera);
this mock replays data/*.png in the same shape so the pipeline runs without a device."""

from pathlib import Path
from typing import Iterator

from src.config import DATA_DIR


def sample_frames(data_dir: Path = DATA_DIR) -> Iterator[tuple[str, bytes]]:
    """Yield (name, image_bytes) for each sample image, sorted by name."""
    for path in sorted(data_dir.glob("*.png")):
        yield path.name, path.read_bytes()
