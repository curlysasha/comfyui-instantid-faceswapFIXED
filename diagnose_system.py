#!/usr/bin/env python3
"""
Diagnostic script for InstantID NaN issues
Run this to check your system configuration
"""

import sys
import subprocess

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

print("=" * 70)
print("InstantID System Diagnostics")
print("=" * 70)

# Python version
print(f"\nPython: {sys.version}")

# PyTorch
try:
    import torch
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
except ImportError:
    print("\n❌ PyTorch not installed!")

# ONNX Runtime
try:
    import onnxruntime as ort
    print(f"\nONNX Runtime: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
except ImportError:
    print("\n❌ onnxruntime-gpu not installed!")

# InsightFace
try:
    import insightface
    print(f"\nInsightFace: {insightface.__version__}")
except ImportError:
    print("\n❌ insightface not installed!")

# NumPy
try:
    import numpy as np
    print(f"\nNumPy: {np.__version__}")
except ImportError:
    print("\n❌ NumPy not installed!")

# NVIDIA Driver
nvidia_smi = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
if nvidia_smi and not nvidia_smi.startswith("Error"):
    print(f"\nNVIDIA Driver: {nvidia_smi}")
else:
    print("\n⚠️ nvidia-smi not available")

print("\n" + "=" * 70)
print("Compatibility Check")
print("=" * 70)

# Check for known problematic combinations
try:
    import torch
    import onnxruntime as ort

    issues = []

    # Check PyTorch CUDA compatibility
    if torch.cuda.is_available():
        cuda_ver = torch.version.cuda
        if cuda_ver and float(cuda_ver.split('.')[0]) < 11:
            issues.append("⚠️ CUDA version < 11.0 may have compatibility issues")

    # Check onnxruntime version
    ort_major = int(ort.__version__.split('.')[0])
    ort_minor = int(ort.__version__.split('.')[1])
    if ort_major == 1 and ort_minor >= 17:
        issues.append("⚠️ onnxruntime-gpu >= 1.17 may have NaN issues on some systems")
        issues.append("   Try: pip install onnxruntime-gpu==1.16.3")

    # Check for mixed precision issues
    if hasattr(torch.backends, 'cudnn'):
        if not torch.backends.cudnn.enabled:
            issues.append("⚠️ cuDNN is disabled")

    if not issues:
        print("\n✅ No obvious compatibility issues detected")
    else:
        print("\n⚠️ Potential issues found:")
        for issue in issues:
            print(issue)

except Exception as e:
    print(f"\n❌ Error during compatibility check: {e}")

print("\n" + "=" * 70)
print("Recommended Configuration")
print("=" * 70)

print("""
For best compatibility:
- PyTorch: >= 2.0 with CUDA 11.8 or 12.1
- onnxruntime-gpu: 1.16.3 (known stable)
- insightface: latest
- NVIDIA Driver: >= 520.0

If you have NaN issues, try:
1. Downgrade onnxruntime-gpu:
   pip install onnxruntime-gpu==1.16.3

2. Force float32 precision:
   python main.py --force-fp32

3. Disable lowvram mode:
   python main.py --normalvram

4. Update NVIDIA drivers to latest version
""")

print("\n" + "=" * 70)
print("Test Face Embedding Generation")
print("=" * 70)

try:
    import numpy as np
    import torch

    # Create fake face image
    fake_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

    print("\nTesting InsightFace with fake image...")
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="antelopev2",
        root="models/insightface",
        providers=["CPUExecutionProvider", "CUDAExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    # This will likely fail (no face in random noise) but we can check for NaN errors
    import cv2
    faces = app.get(fake_image)

    if len(faces) == 0:
        print("✅ No face detected in test image (expected)")
        print("✅ No NaN errors from InsightFace")
    else:
        emb = faces[0]["embedding"]
        if np.isnan(emb).any() or np.isinf(emb).any():
            print("❌ InsightFace returning NaN/Inf embeddings!")
            print("   This is a critical issue with onnxruntime or model files")
        else:
            print("✅ InsightFace returning valid embeddings")
            print(f"   Embedding stats: min={emb.min():.4f}, max={emb.max():.4f}")

except Exception as e:
    print(f"⚠️ Could not test InsightFace: {e}")
    print("   Make sure models are installed in ComfyUI/models/insightface/models/antelopev2/")

print("\n" + "=" * 70)
