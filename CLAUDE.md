# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ComfyUI extension for InstantID-based face swapping. It implements face replacement and generation using InsightFace for face detection/embedding and InstantID adapters for SDXL models. The extension provides custom nodes for ComfyUI workflows.

**IMPORTANT**: This extension works ONLY with SDXL checkpoints.

## Installation & Dependencies

```bash
# Install dependencies
pip install -r requirements.txt

# Dependencies include:
# - insightface (face analysis and embedding)
# - onnxruntime-gpu (accelerated inference)
```

## Required Model Files

The extension requires several model files to be manually downloaded and placed in specific ComfyUI directories:

1. **AntelopeV2 Models** → `ComfyUI/models/insightface/models/antelopev2/`:
   - 1k3d68.onnx, 2d106det.onnx, genderage.onnx, glintr100.onnx, scrfd_10g_bnkps.onnx

2. **InstantID Models** → `ComfyUI/models/ipadapter/` and `ComfyUI/models/controlnet/`:
   - ip-adapter.bin (InstantID adapter)
   - diffusion_pytorch_model.safetensors + config.json (ControlNet)

## Architecture Overview

### Core Components

1. **Custom Nodes** (`nodes.py`): 
   - Main ComfyUI node implementations for face processing pipeline
   - 11 custom nodes for loading models, preprocessing, applying adapters

2. **IP Adapter Module** (`ip_adapter/`):
   - `instantId.py`: InstantID adapter implementation with cross-attention patching
   - `resampler.py`: Face embedding resampler for conditioning

3. **Utilities** (`utils.py`):
   - Image preprocessing (rotation, keypoint drawing, masking)
   - Math utilities for face alignment and geometric transformations

4. **UI Components** (`ui/`):
   - Custom JavaScript for interactive keypoint drawing
   - WebGL shaders for visual effects

### Key Node Types

- **LoadInsightface**: Initializes face detection/analysis
- **FaceEmbed**: Extracts face embeddings from reference images
- **LoadInstantIdAdapter**: Loads InstantID adapter and resampler
- **PreprocessImage**: Prepares pose images with masking and resizing
- **Apply Nodes**: Apply InstantID adapter and ControlNet conditioning
- **KpsMaker**: Interactive keypoint drawing tool
- **Rotation Nodes**: Handle face alignment through image rotation

### Data Flow

1. Load InsightFace → Extract face embeddings from reference image
2. Load InstantID adapter + resampler → Process embeddings for conditioning
3. Preprocess pose image → Extract/align face region with masking
4. Apply InstantID adapter to SDXL model (optional)
5. Apply ControlNet with face conditioning for generation
6. Generate with conditioned model

## Workflow Examples

The `workflows/` directory contains JSON workflow examples:

- `simple.json`: Basic face swap workflow
- `simple_with_adapter.json`: Face swap with InstantID adapter
- `draw_kps.json`: Manual keypoint placement workflow
- `auto_rotate.json`: Automatic face alignment
- `promp2image.json`: Text-to-image with face control
- `inpaint.json`: General inpainting workflow

## Development Notes

### Image Processing
- All images processed at 8-pixel multiples for SDXL compatibility
- Face detection requires minimum mask area and padding
- Keypoint system uses 5-point face landmarks (eyes, nose, mouth corners)

### GPU Memory Management
- Models loaded on demand to conserve VRAM
- Supports manual offload mechanisms
- Compatible with `--fp16-vae` ComfyUI argument

### Error Handling
- "No face detected" errors typically require larger masks or manual keypoints
- Face rotation detection may fail with extreme angles
- InsightFace requires proper ONNX provider configuration

### UI Integration
- Custom nodes register with ComfyUI's node system
- Interactive keypoint editor with drag/zoom controls
- Real-time angle calculation for face alignment

## Common Issues

1. **Black square output**: Fixed in latest version - dtype auto-detection implemented. See [FIX_BLACK_SQUARE.md](FIX_BLACK_SQUARE.md)
2. **No face detected**: Increase mask size or use manual keypoint drawing
3. **Model loading errors**: Verify model file placement and paths
4. **Memory issues**: Use FP16 mode or enable manual offload
5. **Extreme rotation**: Use manual keypoint placement instead of auto-detection

## Recent Fixes

### Black Square Output Fix (2025-11-18)

**Problem**: Black square output in newer ComfyUI versions with InstantID nodes

**Root Cause**: Hardcoded `torch.float16` in InstantID code conflicted with new ComfyUI's `torch.bfloat16` VAE, causing NaN/Inf values.

**Solution Applied**:
- `ip_adapter/instantId.py:32` - Changed from hardcoded `dtype = torch.float16` to auto-detect `dtype = q.dtype`
- `ip_adapter/resampler.py:121` - Added clamping to prevent NaN/Inf propagation: `torch.clamp(output, min=-65504, max=65504)`

**Additional Solutions**: See [FIX_BLACK_SQUARE.md](FIX_BLACK_SQUARE.md) for alternative fixes if auto-dtype doesn't resolve the issue.

**Files Modified**:
- [ip_adapter/instantId.py:32](ip_adapter/instantId.py#L32) - Auto-detect dtype from query tensor
- [ip_adapter/resampler.py:121](ip_adapter/resampler.py#L121) - Clamp output range

### Face Detection Error Handling (2025-07-02)
Fixed critical issues where missing faces would crash ComfyUI workflows:

**Problem**: Assert errors (`"No face detected for face embed"`) would terminate entire workflows

**Solution**: Graceful degradation with full skip of InstantID processing
- `FaceEmbed` → Returns empty tensor instead of crashing
- `FaceEmbedCombine` → Handles empty embeddings with zero conditioning  
- `InstantIdAdapterApply` → Skips adapter when no face detected
- `ControlNetInstantIdApply` → Skips ControlNet when no face detected
- `AngleFromFace` → Returns 0° angle when no face found
- `get_kps_from_image` → Returns None instead of assert error

**Result**: When no face is detected, workflow continues and generates normal images without face swap attempts.

**Files Modified**: `nodes.py` (lines 48-62, 84-96, 244-259, 292-332, 130-142), `utils.py` (lines 97-104)

## Extension Integration

The codebase follows ComfyUI's extension patterns:
- `__init__.py` exports node mappings for ComfyUI registration
- `WEB_DIRECTORY` points to UI assets
- Custom data types (FACE_EMBED, INSIGHTFACE_APP, etc.) for node chaining
- Manual model download required (no auto-download of antelopev2.zip)