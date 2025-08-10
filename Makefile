# Makefile for mm-analytics — one-command E2E via `make test`

# ---- Config ----
IMAGE            ?= mm-analytics
TAG              ?= latest
IMAGE_REF        := $(IMAGE):$(TAG)

PORT             ?= 8000
HF_CACHE_VOLUME  ?= hf_cache
CKPT             ?= artifacts/best_model.pt

# Synthetic data
NUM_SAMPLES      ?= 600
IMAGE_SIZE       ?= 64
SYNTH_DIR        ?= data/synth
DATA_STAMP       := $(SYNTH_DIR)/.done

# Tools & runtime
DOCKER           ?= docker
CURL             ?= curl
CONTAINER_NAME   ?= mm-analytics-api
HEALTH_HOST      ?= 127.0.0.1
APP_PORT         ?= 8000
HEALTH_PATH      ?= /openapi.json
HEALTH_URL       := http://$(HEALTH_HOST):$(PORT)$(HEALTH_PATH)
WAIT_SECS        ?= 600

# Helper used for common run pattern
RUN_BASE         := $(DOCKER) run --rm -v "$(PWD)":/app $(IMAGE_REF)

# Default target
.DEFAULT_GOAL := help

# ---- Phonies ----
.PHONY: help build dataset train serve up down logs wait test-only test quickstart clean clean-data clean-artifacts check-tools check-port kill-port

help:
	@echo ""
	@echo "Targets:"
	@echo "  make test        Build → dataset → train → start API → wait → run curls → stop API (E2E)"
	@echo "  make quickstart  Build → dataset → train → serve (blocking server)"
	@echo "  make build       Build Docker image $(IMAGE_REF)"
	@echo "  make dataset     Generate synthetic dataset into $(SYNTH_DIR)"
	@echo "  make train       Train model (writes $(CKPT))"
	@echo "  make serve       Run API server (blocking)"
	@echo "  make up/down     Start/stop API server (detached)"
	@echo "  make logs        Follow server logs"
	@echo "  make check-port  Check if port $(PORT) is available"
	@echo "  make kill-port   Kill process using port $(PORT)"
	@echo "  make clean*      Remove dataset/artifacts"
	@echo ""
	@echo "Vars: PORT=$(PORT)  NUM_SAMPLES=$(NUM_SAMPLES)  IMAGE_SIZE=$(IMAGE_SIZE)"
	@echo ""
	@echo "If port $(PORT) is in use, either:"
	@echo "  - Run: make kill-port"
	@echo "  - Or use a different port: make test PORT=8001"

# ---- Sanity checks ----
check-tools:
	@command -v $(DOCKER) >/dev/null 2>&1 || { echo "✖ Docker not found in PATH"; exit 1; }
	@command -v $(CURL)   >/dev/null 2>&1 || { echo "✖ curl not found in PATH"; exit 1; }

# ---- Port management ----
check-port:
	@echo "Checking if port $(PORT) is available..."
	@if lsof -Pi :$(PORT) -sTCP:LISTEN -t >/dev/null 2>&1; then \
		echo "✖ Port $(PORT) is already in use by:"; \
		lsof -Pi :$(PORT) -sTCP:LISTEN; \
		echo ""; \
		echo "Options:"; \
		echo "  1. Run 'make kill-port' to kill the process"; \
		echo "  2. Use a different port: 'make test PORT=8001'"; \
		echo "  3. Manually stop the process using port $(PORT)"; \
		exit 1; \
	else \
		echo "✓ Port $(PORT) is available"; \
	fi

kill-port:
	@echo "Attempting to kill process on port $(PORT)..."
	@PID=$$(lsof -Pi :$(PORT) -sTCP:LISTEN -t 2>/dev/null); \
	if [ -n "$$PID" ]; then \
		echo "Found process $$PID using port $(PORT)"; \
		kill -9 $$PID && echo "✓ Killed process $$PID" || echo "✖ Failed to kill process $$PID (may need sudo)"; \
	else \
		echo "No process found using port $(PORT)"; \
	fi

# ---- Build image (only if Dockerfile changed) ----
.docker-build-stamp: Dockerfile
	$(DOCKER) build -t $(IMAGE_REF) .
	@touch .docker-build-stamp

build: check-tools .docker-build-stamp

# ---- Generate dataset (idempotent via stamp) ----
$(DATA_STAMP): .docker-build-stamp
	@if [ ! -f $(DATA_STAMP) ]; then \
		mkdir -p $(SYNTH_DIR); \
		$(RUN_BASE) python scripts/generate_synth_dataset.py \
			--out_dir $(SYNTH_DIR) --num_samples $(NUM_SAMPLES) --image_size $(IMAGE_SIZE); \
		touch $(DATA_STAMP); \
	else \
		echo "Dataset already exists at $(SYNTH_DIR) (delete $(DATA_STAMP) to regenerate)"; \
	fi

dataset: $(DATA_STAMP)

# ---- Train (produces checkpoint) ----
$(CKPT): $(DATA_STAMP) .docker-build-stamp
	@if [ ! -f $(CKPT) ]; then \
		$(DOCKER) run --rm --shm-size=1g \
			-v "$(PWD)":/app -v $(HF_CACHE_VOLUME):/root/.cache/huggingface \
			$(IMAGE_REF) \
			python scripts/train.py --config config/default.yaml; \
		test -f "$(CKPT)" || { echo "✖ Expected checkpoint '$(CKPT)' not found after training." && exit 1; }; \
	else \
		echo "Model checkpoint already exists at $(CKPT) (delete to retrain)"; \
	fi

train: $(CKPT)

# ---- Serve (blocking) ----
serve: .docker-build-stamp $(CKPT) check-port
	$(DOCKER) run --rm \
		-p $(PORT):$(APP_PORT) \
		-v "$(PWD)":/app \
		-e MM_CKPT=$(CKPT) \
		$(IMAGE_REF)

# ---- Detached server management ----
up: .docker-build-stamp $(CKPT) check-port
	@$(DOCKER) rm -f $(CONTAINER_NAME) >/dev/null 2>&1 || true
	$(DOCKER) run -d --name $(CONTAINER_NAME) \
		-p $(PORT):$(APP_PORT) \
		-v "$(PWD)":/app \
		-e MM_CKPT=$(CKPT) \
		$(IMAGE_REF)
	@echo "✓ API starting on http://$(HEALTH_HOST):$(PORT)"

down:
	@$(DOCKER) rm -f $(CONTAINER_NAME) >/dev/null 2>&1 || true
	@echo "✓ API stopped."

logs:
	$(DOCKER) logs -f $(CONTAINER_NAME)

# ---- Wait for API readiness ----
wait:
	@echo "Waiting for API at $(HEALTH_URL) (timeout: $(WAIT_SECS)s) ..."
	@i=0; \
	while [ $$i -lt $(WAIT_SECS) ]; do \
	  if $(CURL) -sf -o /dev/null "$(HEALTH_URL)"; then \
	    echo "✓ API is responsive."; exit 0; \
	  fi; \
	  i=$$((i+1)); sleep 1; \
	done; \
	echo "✖ API did not become ready within $(WAIT_SECS)s"; \
	$(DOCKER) logs --tail=100 $(CONTAINER_NAME) || true; \
	exit 1

# ---- Smoke tests ----
test-only:
	@echo "Running smoke tests..."
	@$(CURL) -X POST http://127.0.0.1:$(PORT)/predict-multipart \
	  -F text="red square low value" \
	  -F 'tabular_json={"num_a":0.1,"num_b":2.0,"cat_x":"A"}' \
	  && echo "✓ Text+tabular test passed" || { echo "✖ Text+tabular test failed"; exit 1; }
	@if [ -f $(SYNTH_DIR)/images/sample_0.png ]; then \
	  $(CURL) -X POST http://127.0.0.1:$(PORT)/predict-multipart \
	    -F image=@$(SYNTH_DIR)/images/sample_0.png \
	    && echo "✓ Image test passed" || { echo "✖ Image test failed"; exit 1; }; \
	else \
	  echo "⚠ No test image found, skipping image test"; \
	fi

# ---- One command: full E2E including cleanup ----
test: build dataset train
	@echo "Starting end-to-end test..."
	@set -e; \
	trap '$(MAKE) --no-print-directory down' EXIT; \
	$(MAKE) --no-print-directory up; \
	$(MAKE) --no-print-directory wait; \
	$(MAKE) --no-print-directory test-only; \
	echo "✅ End-to-end test passed."

# ---- Quickstart (serve and keep it running) ----
quickstart: build dataset train
	@$(MAKE) --no-print-directory serve

# ---- Cleanup ----
clean-data:
	rm -rf $(SYNTH_DIR)

clean-artifacts:
	rm -rf artifacts

clean-docker:
	rm -f .docker-build-stamp

clean: clean-data clean-artifacts clean-docker
	@echo "✓ All cleaned"