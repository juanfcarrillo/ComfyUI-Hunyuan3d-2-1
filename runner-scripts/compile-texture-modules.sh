#!/bin/bash

# Compile custom rasterizer
cd hy3dpaint/custom_rasterizer
python setup.py install

# Compile differentiable renderer
cd ../DifferentiableRenderer
python setup.py install