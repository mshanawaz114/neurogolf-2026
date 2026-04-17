PYTHON := python3
PIP    := pip3

.PHONY: install venv data lint format test validate score view build-all zip safe-zip clean

# ── Setup ─────────────────────────────────────────────────────────────────────
venv:
	$(PYTHON) -m venv .venv
	@echo "Run: source .venv/bin/activate"

install:
	$(PIP) install -r requirements.txt

# Download tasks via Kaggle CLI
data:
	$(PYTHON) scripts/download_arc.py

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	ruff check .
	black --check .
	isort --check-only .

format:
	black .
	isort .

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

# ── Per-task helpers (usage: make validate TASK=001) ──────────────────────────
TASK ?= 001

validate:
	$(PYTHON) utils/validate.py \
		--onnx onnx/task$(TASK).onnx \
		--task tasks/task$(TASK).json

score:
	$(PYTHON) utils/scoring.py --onnx onnx/task$(TASK).onnx

view:
	netron onnx/task$(TASK).onnx

# ── Auto-solve pipeline ───────────────────────────────────────────────────────
solve-fast:
	$(PYTHON) scripts/solve_all.py --no-learned

solve-all:
	$(PYTHON) scripts/solve_all.py

solve-one:
	$(PYTHON) scripts/solve_all.py --task task$(TASK)

build-all:
	@for f in solutions/task*.py; do \
		echo "Building $$f..."; \
		$(PYTHON) $$f; \
	done

# ── Submission ────────────────────────────────────────────────────────────────
zip:
	@echo "Packaging submission.zip..."
	cd onnx && zip ../submission.zip task*.onnx
	@echo "Done: submission.zip"

safe-zip:
	@echo "Building submission_full_safe.zip..."
	$(PYTHON) scripts/build_safe_submission.py
	@echo "Done: submission_full_safe.zip"

static23-zip:
	@echo "Building submission_full_safe.zip with the static23 profile..."
	$(PYTHON) scripts/build_safe_submission.py --profile static23
	@echo "Done: submission_full_safe.zip"

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f submission.zip results.csv
