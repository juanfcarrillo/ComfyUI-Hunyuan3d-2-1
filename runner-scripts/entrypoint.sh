#!/bin/bash
set -e

chmod +x /app/runner-scripts/download.sh
bash /app/runner-scripts/download.sh

chmod +x /app/runner-scripts/compile-texture-modules.sh
bash /app/runner-scripts/compile-texture-modules.sh

echo "########################################"
echo "[INFO] Starting Server..."
echo "########################################"

# Apply PyTorch compatibility patch
echo "[INFO] Applying PyTorch compatibility patch..."
export PYTHONPATH="/app:$PYTHONPATH"

python -c "import sys; sys.path.insert(0, '/app'); import pytorch_patch" 2>/dev/null || echo "[WARN] PyTorch patch not loaded"
python -m http.server 8188 --bind 0.0.0.0
