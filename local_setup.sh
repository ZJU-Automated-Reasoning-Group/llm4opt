#!/bin/bash
set -e

# Create or update virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo "Setup complete! Virtual environment is activated."
echo "To deactivate: deactivate"
echo "To activate again: source venv/bin/activate"
