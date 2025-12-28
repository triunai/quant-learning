#!/bin/bash
set -e

echo "Installing dependencies..."
pip install --upgrade pip
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

# Add extra tools for 'Max CI/CD'
pip install mypy types-requests types-setuptools

echo "Downloading TextBlob corpora..."
python -m textblob.download_corpora

echo "Environment setup complete."
