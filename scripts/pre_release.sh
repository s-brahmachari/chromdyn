#!/usr/bin/env bash
set -e  # stop if any command fails

echo "Running Ruff autofix..."
ruff check . --fix

echo "Running Black formatter..."
black .

echo "Running tests..."
pytest -v

echo "Pre-release check completed successfully!"