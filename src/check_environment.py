"""
Environment Check Script for Computer Vision & AI Setup

This script inspects and verifies the Python environment, including core libraries:
- OpenCV
- NumPy
- PyTorch
- Ultralytics (YOLO)
- CUDA / GPU Availability
"""

import sys

def check_env():
    print("================================")
    print("Computer Vision AI Environment")
    print("================================")
    
    # Python Version
    python_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python:      {python_ver}")
    
    # OpenCV Version
    try:
        import cv2
        cv_ver = cv2.__version__
    except ImportError as e:
        cv_ver = f"NOT INSTALLED ({e})"
    print(f"OpenCV:      {cv_ver}")
    
    # NumPy Version
    try:
        import numpy as np
        np_ver = np.__version__
    except ImportError as e:
        np_ver = f"NOT INSTALLED ({e})"
    print(f"NumPy:       {np_ver}")
    
    # PyTorch Version & CUDA check
    cuda_available = False
    gpu_name = "CPU mode"
    try:
        import torch
        torch_ver = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError as e:
        torch_ver = f"NOT INSTALLED ({e})"
    
    print(f"PyTorch:     {torch_ver}")
    
    # Ultralytics Version
    try:
        import ultralytics
        yolo_ver = ultralytics.__version__
    except ImportError as e:
        yolo_ver = f"NOT INSTALLED ({e})"
    print(f"Ultralytics: {yolo_ver}")
    
    print()
    print(f"CUDA available: {cuda_available}")
    print(f"GPU:            {gpu_name}")
    print("================================")

if __name__ == "__main__":
    check_env()
