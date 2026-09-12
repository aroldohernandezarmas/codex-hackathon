# codex-hackathon

## Frame filtering

The gate compares each frame with the last accepted anchor, compensating small
camera translations and global RGB exposure/color changes before measuring local
Lab differences. Translation must improve matching across the background; lighting
is fitted to the quiet majority. Unexplained changes, including large objects, are
sent immediately. There is no extra frame wait or cooldown. Quiet frames and frames
received while the model is busy do not advance the anchor.

Run `poetry run python -m scripts.benchmark_gate` for a comparison with the original
filter. It writes `DATA_DIR/gate-benchmark.csv` without calling any model. The cases
use synthetic shifts, illumination changes and objects on local PNG samples;
results do not establish real-world savings or optimal thresholds. Real shadows,
clipping, large camera motion and weakly textured scenes may still cause calls.

## Notebooks in VS Code

1. Install dependencies: `poetry install`.
2. Install the Microsoft **Python** and **Jupyter** extensions in VS Code.
3. Open or create an `.ipynb` file.
4. Click **Select Kernel → Python Environments** and pick `.venv/bin/python`
   from this project.

The `ipykernel` kernel is part of the Poetry dev dependencies.
