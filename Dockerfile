# 1. Use the official CUDA 13.3 Devel image with Ubuntu 22.04
FROM nvidia/cuda:13.3.0-devel-ubuntu22.04

# 2. Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# 3. Install essential C++ development tools (g++, make, cmake)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Set environment paths so the shell can find nvcc automatically
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# 5. Define the working directory inside the container
WORKDIR /workspace
