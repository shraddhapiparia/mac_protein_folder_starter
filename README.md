# Mac Protein Folding Starter (MLX)

A lightweight project for running **protein structure prediction locally on Apple Silicon** using MLX-based inference.

This repository provides a simple engineering wrapper around a protein folding model, enabling reproducible runs, structured outputs, and basic benchmarking on macOS.

The goal of this project is not model development but **understanding how modern protein folding models run and what outputs they produce** in a local environment.

---

# Background: Protein Structure Prediction

Proteins are linear chains of amino acids that fold into complex three-dimensional structures. The folded structure determines how a protein functions in biological systems.

Understanding protein structure is important for:

- drug discovery
- enzyme engineering
- understanding genetic disease
- studying molecular biology

Traditionally, protein structures were determined experimentally using:

- X-ray crystallography  
- nuclear magnetic resonance (NMR) spectroscopy  
- cryo-electron microscopy  

These methods are accurate but **time-consuming and expensive**, and many proteins are difficult to study experimentally.

Because of this, predicting protein structure directly from its amino acid sequence has been a major challenge in computational biology.

---

# AlphaFold and Modern Protein Folding Models

AlphaFold, developed by DeepMind, transformed protein structure prediction by applying deep learning to the folding problem.

The model predicts a protein's three-dimensional structure directly from its amino acid sequence by combining:

- **multiple sequence alignment (MSA)** information
- **transformer-based attention networks**
- **pairwise residue interaction modeling**
- **structure prediction modules**

The system learns evolutionary relationships between amino acids and uses those constraints to predict spatial arrangements.

AlphaFold achieved near-experimental accuracy in the **CASP14 protein structure prediction challenge**, demonstrating that machine learning can solve many protein folding problems that previously required experimental methods.

Following AlphaFold, several open implementations have emerged, including:

- OpenFold
- FastFold
- ColabFold

These projects reproduce or extend AlphaFold-style architectures for research and experimentation.

---

# Why Run Protein Folding Locally?

Most protein folding pipelines are designed to run on **Linux systems with NVIDIA GPUs using CUDA**.

However, running them locally can be useful for:

- understanding the inference workflow
- inspecting model outputs
- experimenting with protein sequences
- benchmarking runtime and resource usage
- developing analysis tools around predicted structures

With the rise of Apple Silicon, it has become possible to run machine learning workloads locally using **Metal-accelerated compute backends**.

---

# MLX on Apple Silicon

MLX is a machine learning framework developed by Apple that enables efficient tensor computations on Apple Silicon hardware.

Key characteristics include:

- unified memory architecture
- GPU acceleration through Metal
- optimized tensor operations for Apple hardware

MLX plays a role similar to CUDA kernels on NVIDIA GPUs but is designed for macOS systems.

Using MLX, certain machine learning models can be executed locally on Apple Silicon without requiring CUDA-enabled hardware.

---

# Project Goals

This repository focuses on the **engineering aspects of running protein folding inference**:

- creating a lightweight CLI wrapper around a folding model
- structuring outputs for reproducibility
- running small example protein sequences
- capturing logs and artifacts from prediction runs
- exploring runtime behavior on Apple Silicon

The project is intended for **experimentation and system exploration**, not for training new protein folding models.

---

# Environment Setup

Create a conda environment:

```bash
conda create -n macfold python=3.11 -y
conda activate macfold
```

Install core dependencies:
```bash conda install -c conda-forge git pip jupyterlab matplotlib pandas pyyaml -y
```

Install MLX tools:
```bash pip install mlx mlx-lm
```

Export the environment for reproducibility:
```bash conda env export > environment.yml ```

Project structure:

mac_protein_folder_starter/

├── examples/            # example sequences or JSON inputs

├── scripts/             # shell entrypoints for running predictions

├── src/                 # python wrapper code

├── outputs/             # generated results (not tracked in git)
│   ├── logs/
│   └── structures/

├── notebooks/           # analysis notebooks and runtime plots

├── docs/                # figures used in README or blog posts

├── environment.yml      # conda environment specification
├── README.md
└── .gitignore

Example input:

{
  "seeds": [42],
  "queries": {
    "query_1": {
      "chains": [
        {
          "molecule_type": "protein",
          "chain_ids": ["A"],
          "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANLKS"
        }
      ]
    }
  }
}
--- 

# Running a Prediction

Example workflow:
bash scripts/predict.sh examples/query.json

Outputs will be written to:
outputs/structures/
outputs/logs/

Typical artifacts include:

predicted structure files (.cif or .pdb)

MSA logs

template search results

runtime logs

These artifacts are copied into structured output directories for easier inspection.

Example Analysis

The notebooks directory can contain exploratory analysis and plots such as:

sequence length vs runtime

predicted confidence score distributions

visualization of predicted structures

Example notebooks are stored in:
notebooks/

Reproducibility

The repository tracks the software environment using:

environment.yml

Large generated outputs and datasets are excluded from version control.