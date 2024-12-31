# Makefile for RAG application

# Git hooks setup
setup-hooks:
	git config core.hooksPath .githooks

# Variables
IMAGE_NAME = rag-app
CONTAINER_NAME = rag-container
DEFAULT_PORT = 8000

# Default port can be overridden with PORT=xxxx
PORT ?= $(DEFAULT_PORT)

.PHONY: build stop run setup-hooks

# Build the Docker image
build: setup-hooks
	docker build -t $(IMAGE_NAME) .

# Stop running container
stop:
	-docker stop -t 0 $(CONTAINER_NAME) || true
	-docker rm -f $(CONTAINER_NAME) || true

# Run container with configurable port
run:
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(PORT):8000 \
		-v $(PWD)/documents:/app/documents \
		-v $(PWD)/.chromadb:/app/.chromadb \
		$(IMAGE_NAME)
	@echo "Application running on http://localhost:$(PORT)"
