#!/bin/bash
echo "========================================="
echo "ENVIRONMENT CHECK REPORT"
echo "========================================="

# OS Information
echo -e "\n[OS INFORMATION]"
echo "OS Release: $(lsb_release -d 2>/dev/null | cut -f2 || cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"

# WSL2 Detection
echo -e "\n[WSL2 STATUS]"
if grep -qi microsoft /proc/version; then
    echo "WSL2: YES"
    echo "WSL Kernel: $(uname -r)"
    if [ -f /proc/sys/fs/binfmt_misc/WSLInterop ]; then
        echo "WSL Interop: Enabled"
    fi
else
    echo "WSL2: NO (Native Linux or Container)"
fi

# Python Information
echo -e "\n[PYTHON ENVIRONMENT]"
echo "Python3: $(python3 --version 2>&1)"
echo "Pip3: $(pip3 --version 2>&1 | head -1)"
echo "Python Path: $(which python3)"

# ROCm Detection
echo -e "\n[ROCM INSTALLATION]"
if [ -d /opt/rocm ]; then
    echo "ROCm Base: /opt/rocm"
    for rocm_dir in /opt/rocm*; do
        if [ -d "$rocm_dir" ] && [ "$rocm_dir" != "/opt/rocm" ]; then
            echo "ROCm Version: $(basename $rocm_dir)"
        fi
    done
    if [ -f /opt/rocm/.info/version ]; then
        echo "ROCm Version File: $(cat /opt/rocm/.info/version)"
    fi
else
    echo "ROCm: NOT FOUND"
fi

# Check rocminfo
if command -v rocminfo &> /dev/null; then
    echo "rocminfo: Available"
    rocminfo 2>/dev/null | grep -E "Marketing Name|gfx" | head -4
else
    echo "rocminfo: Not available"
fi

# GPU Device Nodes
echo -e "\n[GPU DEVICE NODES]"
echo "/dev/dxg: $(ls -la /dev/dxg 2>&1 | head -1)"
echo "/dev/kfd: $(ls -la /dev/kfd 2>&1 | head -1)"
echo "/dev/dri: $(ls -la /dev/dri 2>&1 | head -1)"

# PyTorch Information
echo -e "\n[PYTORCH STATUS]"
python3 -c "
try:
    import torch
    print(f'PyTorch Version: {torch.__version__}')
    print(f'CUDA Available: {torch.cuda.is_available()}')
    print(f'CUDA Version: {torch.version.cuda if torch.version.cuda else \"None\"}'
    print(f'HIP Version: {torch.version.hip if hasattr(torch.version, \"hip\") else \"None\"}')
    print(f'Device Count: {torch.cuda.device_count()}')
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print(f'Device Name: {torch.cuda.get_device_name(0)}')
except ImportError:
    print('PyTorch: NOT INSTALLED')
except Exception as e:
    print(f'PyTorch Error: {e}')
" 2>&1

# Check for NVIDIA contamination
echo -e "\n[PACKAGE CONTAMINATION CHECK]"
pip3 list 2>/dev/null | grep -i nvidia | head -5 || echo "No NVIDIA packages found"

# Docker Information
echo -e "\n[DOCKER ENVIRONMENT]"
if [ -f /.dockerenv ]; then
    echo "Running in Docker: YES"
    echo "Container ID: $(hostname)"
else
    echo "Running in Docker: NO"
fi

# Memory Information
echo -e "\n[MEMORY STATUS]"
free -h | head -2

# Environment Variables
echo -e "\n[ROCM ENVIRONMENT VARIABLES]"
echo "HSA_OVERRIDE_GFX_VERSION: ${HSA_OVERRIDE_GFX_VERSION:-not set}"
echo "ROCR_VISIBLE_DEVICES: ${ROCR_VISIBLE_DEVICES:-not set}"
echo "HIP_VISIBLE_DEVICES: ${HIP_VISIBLE_DEVICES:-not set}"
echo "ROCM_PATH: ${ROCM_PATH:-not set}"
echo "HIP_PATH: ${HIP_PATH:-not set}"

echo -e "\n========================================="