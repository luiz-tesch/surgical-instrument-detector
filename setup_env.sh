#!/bin/bash
# Setup script for surgical-instrument-detector
# Run from Git Bash on Windows

set -e

echo "Creating virtual environment..."
python -m venv venv

echo "Activating venv..."
source venv/Scripts/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing PyTorch with CUDA 12.4 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "Installing remaining dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete! Activate with: source venv/Scripts/activate"
echo "Verify GPU: python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\""
