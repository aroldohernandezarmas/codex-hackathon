.PHONY: help install test lint format dev clean notebook

JUPYTER_TOKEN ?= hack-jupyter-token

help: ## Показать справку
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Установить зависимости
	poetry install

test: ## Запустить тесты
	poetry run pytest tests/ -v

lint: ## Проверить код
	poetry run black --check .
	poetry run isort --check-only .
	poetry run mypy src main.py

format: ## Отформатировать код
	poetry run black .
	poetry run isort .

dev: ## Запустить приложение
	poetry run python main.py

notebook: ## Jupyter Lab на :8889 (для MCP)
	@poetry run python -c "import jupyter_server_ydoc" 2>/dev/null || \
		{ echo "нет jupyter-collaboration — read/edit-инструменты MCP будут падать с 404 на /api/collaboration/. Запусти: make install"; exit 1; }
	poetry run jupyter lab --ServerApp.root_dir=$(CURDIR) \
		--IdentityProvider.token=$(JUPYTER_TOKEN) \
		--ServerApp.allow_origin='*' --no-browser --port=8889 \
		--ServerApp.port_retries=0

clean: ## Убрать мусор
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache
