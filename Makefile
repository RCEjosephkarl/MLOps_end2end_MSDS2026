# Convenience shortcuts around docker compose.
# Linux / macOS / WSL / Git Bash. Windows (PowerShell): use make.ps1 instead,
# or run the docker compose commands directly.

.PHONY: help build train serve api gradio mlflow test up down clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

build:  ## Build the shared image
	docker compose build

train:  ## Run the full pipeline (writes models/, reports/, mlruns/)
	docker compose run --rm pipeline

serve:  ## Start API (8000) + Gradio UI (7860)
	docker compose up api gradio

api:  ## Start only the FastAPI server (8000)
	docker compose up api

gradio:  ## Start only the Gradio UI (7860)
	docker compose up gradio

mlflow:  ## Start the MLflow UI (5000)
	docker compose --profile tools up mlflow

test:  ## Run the test suite
	docker compose run --rm test

up:  ## Build + serve (api auto-runs the pipeline first)
	docker compose build && docker compose up api gradio

down:  ## Stop and remove all containers
	docker compose down

clean:  ## down + remove generated artifacts
	docker compose down -v
	rm -rf reports/* mlruns/* models/*.joblib
