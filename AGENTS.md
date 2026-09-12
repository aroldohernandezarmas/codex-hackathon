# AGENTS.md

Instructions for agents (Codex, Claude Code) in this repository.
`CLAUDE.md` is a symlink to this file — edit only `AGENTS.md`.

## Project

Skeleton for now: Python 3.10 + Poetry, entry point `main.py`, code in `src/`, tests in `tests/`.
Deploy: Render (native Python runtime, `render.yaml`), auto-deploy on push to `master`.

## Commands

```bash
make install      # poetry install (venv in ./.venv)
make test         # pytest tests/ -v
make format       # black + isort — ALWAYS after code changes
make lint         # black --check, isort --check-only, mypy
make dev          # poetry run python main.py
```

Deploy: push to `master` — Render builds from `render.yaml`. Secrets go in the Render
dashboard (Environment), never in the repo. Logs and manual redeploys are in the dashboard too.

## Rules

- Dependencies only via `poetry add` — don't edit `pyproject.toml` by hand, and commit `poetry.lock`.
- After changing code: `make format`, then `make lint` and `make test`. Everything must be green.
- Settings come from environment variables in `src/config.py`, defaults are mandatory. Add every new variable to `.env.example`; never commit `.env`.
- Secrets (`.env`, keys, sessions) never go into git.
- Don't multiply abstraction layers or empty packages. A new module appears when there is code that belongs in it.
- Logging via `loguru`, level from `LOG_LEVEL`. No `print`.
- Write data only to `DATA_DIR`. On Render the filesystem is ephemeral — it is wiped on every deploy, so nothing there survives. Anything that must persist goes to an external store.
- **`.ipynb` — only through the `jupyter` MCP.** No Read/Write/Edit/NotebookEdit, no `cat`/`sed`/`jq` on notebook json. Reading, editing and running cells — exclusively via `mcp__jupyter__*` tools.

## Notebooks

Server: `make notebook` — Jupyter Lab on `:8889`, root_dir = repo root, token `JUPYTER_TOKEN` (default `hack-jupyter-token`).
The `jupyter` MCP is registered in the project's local scope and talks to the same port. Its tools appear only after the agent session restarts.

Rule for working with `.ipynb`:

1. Every read and every edit goes through `mcp__jupyter__*`. Direct file access is forbidden: notebook json holds outputs, counters and metadata; editing it by hand breaks them and conflicts with the live kernel.
2. If the `jupyter` MCP is unavailable (no tools, server not responding, calls failing) — **stop**. Don't work around it with file tools. Tell the user the MCP is down and wait for their decision.

The MCP read/edit tools don't use the REST API but the RTC endpoint `/api/collaboration/`,
so the server requires the `jupyter-collaboration` package (dev dependency).
Symptom when it's missing: `use_notebook` and `execute_code` work, but `read_notebook`/`read_cell`
fail with `404 ... /api/collaboration/session/...`. Fix: `make install`, restart `make notebook`.
`make notebook` checks this itself and fails with a hint.
RTC junk (`.jupyter/`, `.jupyter_ystore.db`) is in `.gitignore`.

### How to write to a notebook so the result persists

1. **One cell = one step.** Definitions (imports, config, functions) separate from runs.
   Editing a prompt means re-running one cell, not the whole notebook.
2. **Anything that must persist goes in a cell.** `execute_code` runs in the kernel but writes
   nothing to the file; it's a scratchpad. Measurements and comparisons you cite in your answer
   go into a cell that you run, otherwise the user opens the notebook and finds nothing.
3. **After editing a cell, run it.** Otherwise the file has code without output. Especially
   important before saying "done".
4. **Check that the output was actually saved.** `execute_cell` on long cells sometimes
   returns a result but doesn't put it into the document. Check via `mcp__jupyter__execute_code`
   with `nbformat.read(...)` on the file path: walk `cells` and look at `outputs`.
   Empty — re-run the cell.
5. **Parallelize or split long cells.** Network calls via
   `ThreadPoolExecutor`; a cell that runs for minutes loses output and is painful to wait on.
6. **Checks next to code.** If a cell has logic rather than a one-liner, add a neighboring cell
   with expected values and `assert`. A notebook without checks only measures time.

## Structure

```
main.py            entry point
src/config.py      settings from env
src/               application code
tests/             pytest
render.yaml        Render service: python runtime, poetry install --only main
```
