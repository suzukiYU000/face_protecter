from __future__ import annotations

import platform
import sys

import torch

print("Python:", sys.version.split()[0])
print("Platform:", platform.platform())
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print("CUDA device:", torch.cuda.get_device_name(0))
    except Exception as exc:
        print("CUDA device name error:", exc)

mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
print("MPS available:", mps_available)
if hasattr(torch.backends, "mps"):
    try:
        print("MPS built:", torch.backends.mps.is_built())
    except Exception:
        pass
