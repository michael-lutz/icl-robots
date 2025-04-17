# Makefile

format:
	@echo "=== Running Black ==="
	@black iclrobot tests examples
	@echo ""
	@echo "=== Running Ruff Formatting ==="
	@ruff format iclrobot tests examples
	@echo ""
	@echo "=== Running Ruff Checks ==="
	@ruff check --fix iclrobot tests examples
.PHONY: format

static-checks:
	@echo "=== Running Black ==="
	@black --diff --check iclrobot tests examples
	@echo ""
	@echo "=== Running Ruff Checks ==="
	@ruff check iclrobot tests examples
	@echo ""
	@echo "=== Running MyPy ==="
	@mypy --install-types --non-interactive iclrobot tests examples
.PHONY: lint

test:
	python -m pytest
.PHONY: test
