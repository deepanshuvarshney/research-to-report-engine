.PHONY: install install-embeddings test lint fmt typecheck ci up down logs eval smoke

install:
	python -m pip install -e ".[dev]"

install-embeddings:
	python -m pip install -e ".[dev,embeddings]"

test:
	pytest -q

lint:
	ruff check src tests
	ruff format --check src tests

fmt:
	ruff format src tests
	ruff check --fix src tests

typecheck:
	mypy

ci: lint typecheck test

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

eval:
	python -m research_engine.eval.harness

smoke:
	./scripts/smoke.sh
