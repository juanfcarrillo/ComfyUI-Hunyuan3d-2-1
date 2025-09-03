# ComfyUI with Hunyuan 2.1 Support - Single Stage Build
# Compatible with Runpod and RTX 4090 (CUDA 12.8+)
# Version: 3.0.1

FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_CACHE_DISABLE=0
ENV CUDA_MODULE_LOADING=LAZY
ENV TORCH_LOAD_WEIGHTS_ONLY=False

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    aria2 \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel

# Install ComfyUI dependencies
RUN pip install --no-cache-dir \
    runpod>=1.7.6 \
    Pillow==10.4.0 \
    opencv-python==4.10.0.84 \
    transformers==4.45.2 \
    accelerate==0.34.2 \
    diffusers==0.30.3 \
    xformers \
    einops==0.8.0 \
    omegaconf==2.3.0 \
    safetensors==0.4.5 \
    aiohttp==3.10.8 \
    aiofiles==24.1.0 \
    websockets==13.1 \
    kornia==0.7.3 \
    spandrel==0.3.4 \
    soundfile==0.12.1 \
    matplotlib==3.9.2 \
    numba==0.60.0 \
    scipy==1.14.1 \
    psutil==6.0.0 \
    sentencepiece==0.2.0 \
    protobuf==5.28.2 \
    tokenizers==0.20.0 \
    regex==2024.9.11 \
    torchsde \
    av \
    imageio \
    imageio-ffmpeg \
    scikit-image \
    tqdm \
    pyyaml \
    typing-extensions \
    segment-anything \
    insightface \
    onnxruntime \
    timm \
    rembg \
    ultralytics \
    comfyui-frontend-package \
    nunchaku \
    trimesh \
    pymeshlab \
    pytorch_lightning \
    xatlas \
    pygltflib \
    meshlib \
    pybind11 \
    configargparse \
    huggingface-hub \
    numpy \
    realesrgan \
    basicsr \
    facexlib \
    gfpgan \
    torchvision

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI

# Install ComfyUI's requirements.txt
RUN pip install --no-cache-dir -r /app/ComfyUI/requirements.txt

# Pre-install ComfyUI-Manager  
RUN cd /app/ComfyUI/custom_nodes && \
    git clone --depth=1 --no-tags --recurse-submodules --shallow-submodules \
    https://github.com/ltdrdata/ComfyUI-Manager.git

# Install ComfyUI-Manager requirements
RUN pip install --no-cache-dir -r /app/ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt

# Copy runner scripts
COPY runner-scripts/ /runner-scripts/
RUN chmod +x /runner-scripts/*.sh

# Create directories for models and outputs
RUN mkdir -p /app/ComfyUI/models/checkpoints \
    && mkdir -p /app/ComfyUI/models/vae \
    && mkdir -p /app/ComfyUI/models/clip \
    && mkdir -p /app/ComfyUI/models/unet \
    && mkdir -p /app/ComfyUI/models/diffusion_models \
    && mkdir -p /app/ComfyUI/models/upscale_models \
    && mkdir -p /app/ComfyUI/output \
    && mkdir -p /app/ComfyUI/input \
    && mkdir -p /app/ComfyUI/custom_nodes

# Create PyTorch compatibility patch for weights_only issue
RUN echo "import torch\nimport sys\nimport os\n\n# Monkey patch torch.load to handle weights_only compatibility\noriginal_load = torch.load\n\ndef patched_load(*args, **kwargs):\n    if 'weights_only' not in kwargs:\n        kwargs['weights_only'] = False\n    try:\n        return original_load(*args, **kwargs)\n    except Exception as e:\n        if 'weights_only' in str(e) and kwargs.get('weights_only', False):\n            kwargs['weights_only'] = False\n            return original_load(*args, **kwargs)\n        raise e\n\ntorch.load = patched_load\nprint('PyTorch load compatibility patch applied')" > /app/pytorch_patch.py

# Patch ComfyUI's load_torch_file function directly
RUN sed -i 's/weights_only=True/weights_only=False/g' /app/ComfyUI/comfy/utils.py

# Apply the patch to ComfyUI
RUN echo "import sys\nsys.path.insert(0, '/app')\nimport pytorch_patch" > /app/ComfyUI/pytorch_patch_init.py

# Copy the handler script for Runpod
COPY src/handler.py /app/handler.py

# Set proper permissions
RUN chmod +x /app/handler.py

# Setup volume mount point and working directory
VOLUME /root
WORKDIR /root

# Expose the ComfyUI web interface port
EXPOSE 8188

# Set environment variable for CLI args
ENV CLI_ARGS=""

# Health check to ensure ComfyUI is responsive
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8188/ || exit 1

# Use the enhanced entrypoint
CMD ["bash", "/runner-scripts/entrypoint.sh"]
