#!/bin/bash
set -e

# Install ComfyUI
cd /root
if [ ! -f "/root/.download-complete" ] ; then
    chmod +x /runner-scripts/download.sh
    bash /runner-scripts/download.sh
fi ;

# Compile custom texture modules
cd /root
if [ ! -f "/root/user-scripts/compile-texture-modules.sh" ] ; then
    mkdir -p /root/user-scripts
    cp /runner-scripts/compile-texture-modules.sh /root/user-scripts/compile-texture-modules.sh
    chmod +x /root/user-scripts/compile-texture-modules.sh
    bash /root/user-scripts/compile-texture-modules.sh
else
    echo "[INFO] Running compile-texture-modules script..."
    chmod +x /root/user-scripts/compile-texture-modules.sh
    bash /root/user-scripts/compile-texture-modules.sh
fi

echo "########################################"
echo "[INFO] Starting ComfyUI..."
echo "########################################"

# Apply PyTorch compatibility patch
echo "[INFO] Applying PyTorch compatibility patch..."
export PYTHONPATH="/app:$PYTHONPATH"

cd /root/ComfyUI
python -c "import sys; sys.path.insert(0, '/app'); import pytorch_patch" 2>/dev/null || echo "[WARN] PyTorch patch not loaded"
python main.py --listen 0.0.0.0 --port 8188 ${CLI_ARGS}
