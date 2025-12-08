# ---- Config -----------------------------------------------------------------
UV        ?= uv
PY        ?= python3.13
VENV_DIR  ?= .venv
PYTHON    := $(VENV_DIR)/bin/python

# ---- Phonies ----------------------------------------------------------------
.PHONY: help venv install install-dev format lint test test-real-model clean clean-venv

# ---- Help -------------------------------------------------------------------
help:
	@echo "Targets:"
	@echo "  venv         - create .venv with uv"
	@echo "  install      - install project in editable mode with dev dependencies"
	@echo "  format       - run ruff format"
	@echo "  lint         - run ruff check --fix"
	@echo "  test         - run CPU simulation/correctness tests"
	@echo "  clean        - remove __pycache__, logs, and temporary files"
	@echo "  clean-venv   - remove .venv"

# ---- Environment -------------------------------------------------------------
venv:
	$(UV) venv --python $(PY)
	@echo "Created venv in $(VENV_DIR)"

# Install project + dev deps (ruff, pytest, torch)
install:
	$(UV) pip install --upgrade pip
	$(UV) pip install -e ".[dev]"

# ---- Quality Code ------------------------------------------------------------
format:
	$(UV) run ruff format .

lint:
	$(UV) run ruff check . --fix

# ---- Testing -----------------------------------------------------------------
# Runs the CPU correctness test suite (Stale Pointer Bug check)
# We export CUDA_VISIBLE_DEVICES= to force CPU even if running on a GPU machine
test:
	export CUDA_VISIBLE_DEVICES= && $(UV) run python tests/test_cpu_moe.py

# ---- Housekeeping ------------------------------------------------------------
clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + || true
	find . -name "*.pyc" -delete
	rm -rf logs logs_async logs_sync .pytest_cache .ruff_cache || true

clean-venv:
	rm -rf $(VENV_DIR)
