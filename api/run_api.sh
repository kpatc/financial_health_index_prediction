#!/bin/bash
# Run FastAPI server for Financial Health Index predictions

set -e

echo "Starting Financial Health Index API..."
echo ""

# Activate virtual environment
source ../venv/bin/activate

# Run API with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# API will be available at:
# http://localhost:8000
# API docs: http://localhost:8000/docs
# OpenAPI schema: http://localhost:8000/openapi.json
