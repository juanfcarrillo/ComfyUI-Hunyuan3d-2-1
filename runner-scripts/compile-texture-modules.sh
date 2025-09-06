#!/bin/bash

# Compile custom rasterizer
pip install wheel/*.whl

# Compile differentiable renderer
cd hy3dpaint/DifferentiableRenderer
python setup.py install