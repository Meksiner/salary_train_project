default: help

install:
	uv sync

lint:
	uv run ruff check app src

format:
	uv run ruff format app src

typecheck:
	uv run mypy app src

test:
	uv run pytest

pre-commit:
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push
