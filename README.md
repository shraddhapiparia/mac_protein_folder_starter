# Mac Protein Folding Starter (MLX)

A **Mac-native, reproducible protein folding inference pipeline** built on top of OpenFold3-MLX, designed to make large protein folding models usable on **Apple Silicon** without CUDA. This project focuses on **engineering, reproducibility, and evaluation**, not model training.

## This project has

- A clean **CLI wrapper** around OpenFold3-MLX
- One-command setup and inference on Apple Silicon
- Simple benchmarking for sequence length vs runtime

State-of-the-art protein folding models are typically locked behind CUDA, Linux-only environments. This project demonstrates how modern protein folding models can run locally on macOS and packaged into a usable tool and can be further evaluated for reproducibly.

## What are MLX kernels?

MLX kernels are optimized low-level tensor operations used by Apple’s MLX framework to execute machine-learning workloads efficiently on Apple Silicon. They replace CUDA kernels on macOS by targeting Metal and unified memory, allowing large models to run without explicit CPU <--> GPU data transfers. In protein folding models, MLX kernels power performance-critical operations such as attention, tensor contractions, and normalization.

## Troubleshooting
chmod +x $(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent/'bin'/'torch_shm_manager')")

xattr -dr com.apple.quarantine $(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent/'bin'/'torch_shm_manager')")

## Run
python -m macfold.cli fold --fasta examples/tiny.fasta --out results/run_name
