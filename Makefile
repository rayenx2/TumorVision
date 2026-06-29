.PHONY: setup install test lint format clean train evaluate run-api run-app docker-up

# Setup
setup:
	cp .env.example .env
	@echo "Project structure created. Fill in your .env file."

install:
	pip install -r requirements-dev.txt
	pre-commit install
	pip install -e .
	@echo "All dependencies installed."

# Code quality
lint:
	flake8 src/ api/ app/ tests/ --max-line-length=100 --extend-ignore=E203

format:
	black src/ api/ app/ tests/ --line-length=100
	isort src/ api/ app/ tests/

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# ML Pipeline
train:
	python main.py

train-full:
	RUN_TRAINING=true python main.py

# Running
run-api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

run-app:
	streamlit run app/streamlit_app.py

# Docker
docker-up:
	docker-compose -f docker/docker-compose.yml up --build

docker-down:
	docker-compose -f docker/docker-compose.yml down

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov/
	@echo "Cleaned up."
