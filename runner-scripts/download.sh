#!/bin/bash
set -euo pipefail

echo "########################################"
echo "[INFO] Downloading Models..."
echo "########################################"

# Models

if [ ! -d "/root/app/models" ] ; then
  mkdir -p /root/app/models
fi

aria2c \
  --input-file=/app/runner-scripts/download-models.txt \
  --allow-overwrite=false \
  --auto-file-renaming=false \
  --continue=true \
  --max-connection-per-server=5

# Finish
touch /app/.download-complete
