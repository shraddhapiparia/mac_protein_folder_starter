#!/usr/bin/env bash
set -euo pipefail

python src/cli.py fold \
  --fasta examples/tiny.fasta \
  --out outputs/example_run \
  --upstream openfold-3-mlx