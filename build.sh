#!/bin/bash

# Build script for ComfyUI with Hunyuan 2.1 support
# Version: v4.0.0 - MEGAPAK with ComfyUI-Manager and 25+ custom nodes

IMAGE_NAME="juanfi4/hunyuan21"
VERSION="v4.1.9"

echo "Building Docker image: ${IMAGE_NAME}:${VERSION}"
echo "Creating MEGAPAK version with ComfyUI-Manager and custom nodes..."

# Build the image
docker build --platform linux/amd64 -t "${IMAGE_NAME}:${VERSION}" -t "${IMAGE_NAME}:latest" .

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    
    # Ask if user wants to test locally
    read -p "Do you want to test the ComfyUI interface locally? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Step 2: Testing ComfyUI interface locally..."
        echo "Starting ComfyUI on port 8188..."
        echo "You can access ComfyUI at http://localhost:8188"
        echo "Press Ctrl+C to stop the test"
        docker run -it --rm --gpus all -p 8188:8188 "${IMAGE_NAME}:${VERSION}"
    fi
    
    echo ""
    echo "Next steps:"
    echo "1. Push to Docker Hub:"
    echo "   docker login"
    echo "   docker push ${IMAGE_NAME}:${VERSION}"
    echo ""
    echo "2. Deploy on Runpod:"
    echo "   - Go to https://console.runpod.io/serverless"
    echo "   - Create new endpoint"
    echo "   - Use image: docker.io/${IMAGE_NAME}:${VERSION}"
    echo ""
    
else
    echo "❌ Build failed!"
    exit 1
fi
