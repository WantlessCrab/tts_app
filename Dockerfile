# 1. THE BASE IMAGE

FROM rocm/pytorch:rocm6.4.2_ubuntu24.04_py3.12_pytorch_release_2.6.0

# ----------------------------------------------------------------------
# 2. CORE STACK
# ----------------------------------------------------------------------

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# --- PyTorch & Dependencies  ---

# 1. Downgrade numpy, per dev guidance for this stack
RUN pip3 install --no-cache-dir numpy==1.26.4

# 2. Install PyTorch using 'rocm6.4' wheel URL.
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.4 \
    --break-system-packages

# ----------------------------------------------------------------------
# 3. BASE LAYER CONFIG
# ----------------------------------------------------------------------

WORKDIR /app