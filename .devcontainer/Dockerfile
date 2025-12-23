FROM rocm/jax:latest

ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV ROAM_OVERRIDE_GFX_VERSION=11.0.0

# Install system packages
# Added: gfortran, blas, lapack (Critical for building PETSc)
# Added: libdw-dev (Required for JAX ROCm plugin)
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    libglu1-mesa \
    libgl1 \
    libxrender1 \
    libxcursor1 \
    libxft2 \
    libxinerama1 \
    gfortran \
    libblas-dev \
    liblapack-dev \
    pkg-config \
    libdw-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace