#!/usr/bin/env python3
"""
CrachoLM System Diagnostic & Hardware Verification Script
==========================================================
Verifies Python version, PyTorch installation, CUDA GPU availability,
GPU device name, total VRAM, and tests basic PyTorch tensor creation.
"""

import sys
import os


def run_system_check():
    print("=" * 68)
    print("           CrachoLM System & Hardware Verification")
    print("=" * 68)
    
    # 1. Python Environment Check
    py_ver = sys.version.split()[0]
    print(f"[✓] Python Executable : {sys.executable}")
    print(f"[✓] Python Version    : {py_ver}")
    if sys.version_info < (3, 8):
        print("[!] WARNING: Python 3.8+ is recommended.")
    
    print("-" * 68)

    # 2. PyTorch Import & Version Check
    try:
        import torch
    except ImportError:
        print("[X] ERROR: PyTorch is not installed in this Python environment.")
        print("\nTo install PyTorch with CUDA support on Linux, run:")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("Or for CPU-only mode:")
        print("    pip install torch")
        sys.exit(1)
        
    print(f"[✓] PyTorch Version   : {torch.__version__}")

    # 3. CUDA & GPU Hardware Verification
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        cuda_ver = torch.version.cuda
        
        print(f"[✓] CUDA Available    : YES")
        print(f"[✓] CUDA Version      : {cuda_ver}")
        print(f"[✓] GPU Count         : {device_count}")
        print(f"[✓] GPU Name          : {gpu_name}")
        print(f"[✓] Total VRAM        : {vram_gb:.2f} GB ({vram_bytes:,} bytes)")
        
        # Memory stats
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        print(f"[✓] Currently Allocated: {allocated:.2f} GB")
        print(f"[✓] Currently Reserved : {reserved:.2f} GB")
    else:
        print(f"[!] CUDA Available    : NO (Falling back to CPU)")
        print("    Note: Running on CPU is fine for small testing, but GPU is recommended for training.")

    print("-" * 68)

    # 4. Target Device Automatic Selection Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Compute Device selected by CrachoLM: {device.type.upper()}")

    # 5. PyTorch Tensor Computation Test
    try:
        print(f"[*] Testing tensor allocation & matrix multiplication on [{device.type}]...")
        x = torch.ones((1000, 1000), device=device)
        y = torch.matmul(x, x)
        print(f"[✓] Tensor Test SUCCESS! Result tensor shape: {y.shape}, Device: {y.device}")
    except Exception as e:
        print(f"[X] ERROR performing tensor computation on {device}: {e}")
        sys.exit(1)

    print("=" * 68)
    print(" System check complete! CrachoLM Phase 1 environment is ready.")
    print("=" * 68)


if __name__ == "__main__":
    run_system_check()
