#!/bin/bash
# Oracle VM setup for x14 c14a scanner
# Target: Oracle Cloud VM (152.70.145.41, Oracle Linux 9, 503MB RAM)
#
# Before running: create ~/x14/.env with your credentials (see .env.example)
set -euo pipefail

echo "=== x14 c14a Oracle VM Setup ==="

# System packages
echo "Installing system packages..."
sudo dnf install -y python3.11 python3.11-pip git

# Clone repo (skip if already present)
if [ ! -d ~/x14 ]; then
    echo "Cloning repo..."
    git clone https://github.com/frankywashere/autotrade2.git ~/x14
fi
cd ~/x14
git checkout c14a
git pull

# Ensure surfer_models directory exists (models must be copied manually via scp)
mkdir -p ~/x14/surfer_models

# Python deps
echo "Installing Python dependencies..."
pip3.11 install --user -r v15/panel_dashboard/requirements.txt

# Environment variables for interactive shell
if ! grep -q 'PYTHONPATH=~/x14' ~/.bashrc; then
    echo "Adding PYTHONPATH to ~/.bashrc..."
    cat >> ~/.bashrc << 'ENVEOF'

# x14 scanner environment
export PYTHONPATH=~/x14
export PYTHONUNBUFFERED=1
# Source credentials from .env
set -a; source ~/x14/.env 2>/dev/null; set +a
ENVEOF
fi

# Check .env exists
if [ ! -f ~/x14/.env ]; then
    echo ""
    echo "ERROR: ~/x14/.env not found!"
    echo "Create it with your credentials before starting the service."
    echo "See v15/panel_dashboard/.env.example for the required format."
    exit 1
fi

# Firewall: open port 7860
echo "Opening port 7860 in firewall..."
sudo firewall-cmd --permanent --add-port=7860/tcp
sudo firewall-cmd --reload

# Install systemd service
echo "Installing systemd service..."
sudo cp v15/panel_dashboard/x14-scanner.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable x14-scanner

echo ""
echo "=== Setup complete ==="
echo ""
echo "ML models must be copied manually:"
echo "  scp surfer_models/gbt_model.pkl opc@152.70.145.41:~/x14/surfer_models/"
echo "  scp surfer_models/intraday_ml_model.pkl opc@152.70.145.41:~/x14/surfer_models/"
echo ""
echo "Also open port 7860 in Oracle Cloud VCN Security List (console.oracle.com)"
echo ""
echo "Start service:"
echo "  sudo systemctl start x14-scanner"
echo "  journalctl -u x14-scanner -f"
