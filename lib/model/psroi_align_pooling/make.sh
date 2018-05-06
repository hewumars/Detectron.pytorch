#!/usr/bin/env bash
CUDA_PATH=/usr/local/cuda/

cd src
echo "Compiling psroi_align_pooling kernels by nvcc..."
nvcc -c -o psroi_align_pooling_kernel.cu.o psroi_align_pooling_kernel.cu.cc -x cu -Xcompiler -fPIC -arch=sm_60

cd ../
python build.py
