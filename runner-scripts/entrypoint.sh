#!/bin/bash
set -e

chmod +x /app/runner-scripts/compile-texture-modules.sh
bash /app/runner-scripts/compile-texture-modules.sh

echo "########################################"
echo "[INFO] Starting Server..."
echo "########################################"

# Apply PyTorch compatibility patch
echo "[INFO] Applying PyTorch compatibility patch..."
export PYTHONPATH="/app:$PYTHONPATH"

python runpod_handler.py

