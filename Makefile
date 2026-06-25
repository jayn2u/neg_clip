install: ## [Local development] Upgrade pip, install requirements, install package.
	uv sync

install-dev: ## [Local development] Install test requirements
	uv sync --all-extras

test: ## [Local development] Run unit tests
	python -m pytest -x -s -v tests
