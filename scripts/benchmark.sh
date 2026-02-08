#!/usr/bin/env bash
set -euo pipefail

OUT="results/bench_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

echo "Benchmark output: $OUT"

python -m macfold.cli fold --fasta examples/tiny.fasta --out "$OUT/run_tiny"
echo ""
echo "Done."
