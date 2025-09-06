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
    insightface \
    transparent-background

# Copy project
COPY . .
RUN chmod +x /app/runner-scripts/*.sh

# Use the enhanced entrypoint
CMD ["bash", "/app/runner-scripts/entrypoint.sh"]
