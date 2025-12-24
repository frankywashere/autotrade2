#!/bin/bash

# setup.sh - Run this on your GCE instance to prepare the environment

echo "--- Starting GCE Environment Setup ---"

# 1. Update and install system dependencies
sudo apt-get update
sudo apt-get install -y git htop screen libgl1-mesa-glx libgomp1

# 2. Setup Python environment
# We use 'venv' to keep the system Python clean
python3 -m venv training_env
source training_env/bin/activate

# 3. Install PyTorch with CUDA support
# Using CUDA 12.1 as it is standard for newer Google Deep Learning VMs
echo "Installing PyTorch..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install Project Requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements.txt..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found in current directory."
fi

# 5. Create necessary directories for the hierarchical training pipeline
mkdir -p data/feature_cache
mkdir -p models
mkdir -p logs
mkdir -p results

# 6. Verify GPU and Libraries
echo "--- Verification ---"
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); import ncps; print('ncps library loaded successfully.')"

echo "Setup complete. Source your environment with: source training_env/bin/activate"
