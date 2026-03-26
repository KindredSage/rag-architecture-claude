.PHONY: help install dev run test lint clean docker docker-multi

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -r requirements.txt

dev: ## Install and run in development mode
	pip install -r requirements.txt
	cp -n .env.example .env 2>/dev/null || true
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

run: ## Run in production mode (4 workers)
	gunicorn main:app -k uvicorn.workers.UvicornWorker -w 4 --bind 0.0.0.0:8000 --timeout 120

test: ## Run all tests
	pytest -v --tb=short

test-cov: ## Run tests with coverage report
	pytest --cov=. --cov-report=term-missing --cov-report=html -v

test-safety: ## Run only SQL safety tests
	pytest tests/test_clickhouse.py -v

test-integration: ## Run integration tests
	pytest tests/test_integration.py -v

infra-up: ## Start PostgreSQL + Redis (Docker)
	docker compose up postgres redis -d

infra-down: ## Stop infrastructure
	docker compose down

docker: ## Build and run single-server Docker stack
	docker compose --profile full up --build

docker-multi: ## Build and run multi-server stack with nginx
	docker compose --profile multi-server up --build

clean: ## Clean caches and artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov .coverage /tmp/agent_artifacts /tmp/test_agent_artifacts
