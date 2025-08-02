# CrowdGuard Pro Makefile
# Provides convenient commands for development and deployment

.PHONY: help install dev test lint format clean docker run deploy backup upgrade

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m

# Configuration
PYTHON := python3
UV := uv
STREAMLIT_PORT := 8501
DOCKER_IMAGE := crowdguard-pro
DOCKER_TAG := latest

help: ## Show this help message
	@echo "$(BLUE)CrowdGuard Pro - Development Commands$(NC)"
	@echo "======================================"
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make $(GREEN)<target>$(NC)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Environment Setup

install: ## Install dependencies and set up environment
	@echo "$(BLUE)Installing CrowdGuard Pro...$(NC)"
	@chmod +x setup.sh
	@./setup.sh

dev-install: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) pip install -e ".[dev]"; \
	else \
		$(PYTHON) -m pip install -e ".[dev]"; \
	fi

venv: ## Create virtual environment
	@echo "$(BLUE)Creating virtual environment...$(NC)"
	@$(PYTHON) -m venv .venv
	@echo "$(GREEN)Virtual environment created. Activate with: source .venv/bin/activate$(NC)"

##@ Development

dev: ## Start development server
	@echo "$(BLUE)Starting development server...$(NC)"
	@source .venv/bin/activate && streamlit run app.py --server.port $(STREAMLIT_PORT)

run: ## Run the application
	@echo "$(BLUE)Starting CrowdGuard Pro...$(NC)"
	@source .venv/bin/activate && $(PYTHON) crowdguard.py server start

debug: ## Run application in debug mode
	@echo "$(BLUE)Starting in debug mode...$(NC)"
	@source .venv/bin/activate && $(PYTHON) crowdguard.py server start --dev

##@ Testing

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	@source .venv/bin/activate && python -m pytest tests/ -v

test-cov: ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@source .venv/bin/activate && python -m pytest tests/ --cov=. --cov-report=html --cov-report=term

test-fast: ## Run fast tests only
	@echo "$(BLUE)Running fast tests...$(NC)"
	@source .venv/bin/activate && python -m pytest tests/ -m "not slow" -v

##@ Code Quality

lint: ## Run all linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	@source .venv/bin/activate && \
		flake8 . && \
		black --check . && \
		isort --check-only . && \
		mypy . --ignore-missing-imports

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	@source .venv/bin/activate && \
		black . && \
		isort .

security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	@source .venv/bin/activate && \
		safety check && \
		bandit -r . -f screen

pre-commit: ## Run pre-commit hooks
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	@source .venv/bin/activate && pre-commit run --all-files

##@ Database

db-init: ## Initialize database
	@echo "$(BLUE)Initializing database...$(NC)"
	@source .venv/bin/activate && $(PYTHON) crowdguard.py database init

db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	@source .venv/bin/activate && $(PYTHON) crowdguard.py database migrate

db-backup: ## Create database backup
	@echo "$(BLUE)Creating database backup...$(NC)"
	@source .venv/bin/activate && $(PYTHON) crowdguard.py database backup

##@ Docker

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	@docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	@docker run -p $(STREAMLIT_PORT):$(STREAMLIT_PORT) \
		-v $(PWD)/uploads:/app/uploads \
		-v $(PWD)/logs:/app/logs \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-compose-up: ## Start all services with docker-compose
	@echo "$(BLUE)Starting services with docker-compose...$(NC)"
	@docker-compose up -d

docker-compose-down: ## Stop all services
	@echo "$(BLUE)Stopping services...$(NC)"
	@docker-compose down

docker-logs: ## Show container logs
	@docker-compose logs -f crowdguard-app

##@ Deployment

deploy-staging: ## Deploy to staging
	@echo "$(BLUE)Deploying to staging...$(NC)"
	@# Add staging deployment commands here
	@echo "$(YELLOW)Staging deployment commands would go here$(NC)"

deploy-prod: ## Deploy to production
	@echo "$(BLUE)Deploying to production...$(NC)"
	@# Add production deployment commands here
	@echo "$(YELLOW)Production deployment commands would go here$(NC)"

##@ Management

backup: ## Create application backup
	@echo "$(BLUE)Creating application backup...$(NC)"
	@source .venv/bin/activate && $(PYTHON) crowdguard.py backup create

backup-list: ## List available backups
	@echo "$(BLUE)Available backups:$(NC)"
	@source .venv/bin/activate && $(PYTHON) crowdguard.py backup list

upgrade: ## Upgrade application
	@echo "$(BLUE)Checking for upgrades...$(NC)"
	@source .venv/bin/activate && $(PYTHON) crowdguard.py version upgrade --check-only

status: ## Check application status
	@echo "$(BLUE)Checking application status...$(NC)"
	@source .venv/bin/activate && $(PYTHON) crowdguard.py server status

doctor: ## Run system diagnostics
	@echo "$(BLUE)Running diagnostics...$(NC)"
	@source .venv/bin/activate && $(PYTHON) crowdguard.py doctor

config: ## Show current configuration
	@echo "$(BLUE)Current configuration:$(NC)"
	@source .venv/bin/activate && $(PYTHON) crowdguard.py config show

version: ## Show version information
	@source .venv/bin/activate && $(PYTHON) crowdguard.py version show

##@ Cleanup

clean: ## Clean up temporary files
	@echo "$(BLUE)Cleaning up...$(NC)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .coverage htmlcov/ .pytest_cache/ dist/ build/
	@rm -rf temp/* uploads/* logs/*
	@echo "$(GREEN)Cleanup completed$(NC)"

clean-docker: ## Clean up Docker resources
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	@docker system prune -f
	@docker image prune -f

clean-all: clean clean-docker ## Clean everything

##@ Documentation

docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@# Add documentation generation commands here
	@echo "$(YELLOW)Documentation generation would go here$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	@# Add documentation serving commands here
	@echo "$(YELLOW)Documentation serving would go here$(NC)"

##@ Utilities

logs: ## Show application logs
	@tail -f logs/app.log 2>/dev/null || echo "$(YELLOW)No log file found$(NC)"

monitor: ## Monitor system resources
	@echo "$(BLUE)Monitoring system resources...$(NC)"
	@watch -n 2 'ps aux | grep -E "(streamlit|python)" | grep -v grep; echo ""; df -h; echo ""; free -h'

requirements: ## Update requirements file
	@echo "$(BLUE)Updating requirements...$(NC)"
	@source .venv/bin/activate && \
		if command -v $(UV) >/dev/null 2>&1; then \
			$(UV) pip compile pyproject.toml -o requirements.txt; \
		else \
			pip freeze > requirements.txt; \
		fi

##@ Quick Start

quick-start: install db-init ## Quick start for new users
	@echo "$(GREEN)CrowdGuard Pro setup completed!$(NC)"
	@echo ""
	@echo "$(BLUE)Next steps:$(NC)"
	@echo "1. Edit .env file with your configuration"
	@echo "2. Run: make dev (to start development server)"
	@echo "3. Visit: http://localhost:$(STREAMLIT_PORT)"

demo: ## Run with demo data
	@echo "$(BLUE)Starting with demo data...$(NC)"
	@# Add demo data setup and run commands
	@echo "$(YELLOW)Demo mode would start here$(NC)"
