# Mac Protein Folding Starter (MLX)

A **Mac-native protein folding inference setup** built on top of OpenFold3-MLX, aimed at making large protein folding models runnable on **Apple Silicon** without CUDA. This project focuses on **engineering setup, reproducibility, and system-level evaluation**, not model training.

## This project includes

- A lightweight **CLI wrapper** around OpenFold3-MLX
- One-command setup and inference on Apple Silicon
- Basic benchmarking of sequence length versus runtime

State-of-the-art protein folding models are typically developed and optimized for CUDA-based Linux environments. This repository documents a practical path to running such models locally on macOS for experimentation, inspection of outputs, and understanding system constraints.

## What are MLX kernels?

MLX kernels are optimized low-level tensor operations provided by Apple’s MLX framework to execute machine learning workloads efficiently on Apple Silicon. They serve a role analogous to CUDA kernels on NVIDIA GPUs, targeting Metal and unified memory.

In protein folding models, MLX kernels support performance-critical operations such as attention, tensor contractions, and normalization, enabling inference without explicit CPU–GPU data transfers.

## Hardware notes

- OpenFold is primarily optimized for CUDA-enabled GPUs.
- This repository documents a local Apple Silicon run using the MPS backend for learning and experimentation.
- MPS runs are suitable for small proteins and exploratory analysis, but may be slower and more memory-limited than CUDA-based setups.

## Troubleshooting

```bash
chmod +x $(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent/'bin'/'torch_shm_manager')")
xattr -dr com.apple.quarantine $(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent/'bin'/'torch_shm_manager')")
```
