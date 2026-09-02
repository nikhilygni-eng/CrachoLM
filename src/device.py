"""
CrachoLM Device Detection Module
=================================
Automated hardware detection and management for PyTorch execution.
Detects NVIDIA CUDA GPU availability and safely falls back to CPU if unavailable.
"""

import sys
import torch


def get_device(requested_device: str = "auto") -> torch.device:
    """
    Returns the appropriate PyTorch compute device.
    
    Args:
        requested_device (str): Option to force 'cuda', 'cpu', or 'auto' (default).
        
    Returns:
        torch.device: The selected PyTorch device object.
    """
    if requested_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("[Warning] CUDA was explicitly requested but is not available on this system. Falling back to CPU.")
            return torch.device("cpu")
    elif requested_device == "cpu":
        return torch.device("cpu")
    
    # Auto selection
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info() -> dict:
    """
    Gathers detailed hardware info regarding compute capabilities and VRAM.
    
    Returns:
        dict: Summary of device type, name, CUDA version, total VRAM, etc.
    """
    info = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_type": "cuda" if torch.cuda.is_available() else "cpu",
        "device_name": "CPU",
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "total_vram_gb": 0.0,
        "allocated_vram_gb": 0.0,
        "reserved_vram_gb": 0.0,
    }

    if info["cuda_available"]:
        gpu_props = torch.cuda.get_device_properties(0)
        info["device_name"] = gpu_props.name
        info["total_vram_gb"] = round(gpu_props.total_memory / (1024 ** 3), 2)
        info["allocated_vram_gb"] = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
        info["reserved_vram_gb"] = round(torch.cuda.memory_reserved(0) / (1024 ** 3), 2)

    return info


def print_device_info():
    """
    Displays human-readable summary of device hardware details to stdout.
    """
    info = get_device_info()
    print("=" * 60)
    print("           CrachoLM Hardware & Device Summary           ")
    print("=" * 60)
    print(f" Python Version   : {info['python_version']}")
    print(f" PyTorch Version  : {info['torch_version']}")
    print(f" CUDA Available   : {info['cuda_available']}")
    if info["cuda_available"]:
        print(f" CUDA Version     : {info['cuda_version']}")
        print(f" GPU Count        : {info['device_count']}")
        print(f" GPU Name         : {info['device_name']}")
        print(f" Total VRAM       : {info['total_vram_gb']} GB")
        print(f" Allocated VRAM   : {info['allocated_vram_gb']} GB")
        print(f" Reserved VRAM    : {info['reserved_vram_gb']} GB")
    else:
        print(" Active Device    : CPU (No CUDA-compatible GPU detected)")
    print("=" * 60)


if __name__ == "__main__":
    print_device_info()
    device = get_device()
    print(f"Default target device selected: {device}")
