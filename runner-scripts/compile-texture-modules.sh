#!/bin/bash

# Compile custom rasterizer
pip install hy3dpaint/custom_rasterizer/dist/custom_rasterizer-0.1-cp311-cp311-linux_x86_64.whl

# Compile differentiable renderer
cd hy3dpaint/DifferentiableRenderer
python setup.py install