#!/usr/bin/env bash
set -euo pipefail

# --- basic checks ---
ARCH="$(uname -m)"
if [[ "${ARCH}" != "arm64" ]]; then
  echo "ERROR: Intended for Apple Silicon (arm64). Detected: ${ARCH}"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found."
  exit 1
fi

python3 -m pip install -U pip

# --- clone upstream ---
if [[ ! -d "openfold-3-mlx" ]]; then
  echo "Cloning openfold-3-mlx..."
  git clone https://github.com/latent-spacecraft/openfold-3-mlx.git
else
  echo "openfold-3-mlx already present."
fi

# --- install this wrapper ---
echo "Installing wrapper (editable)..."
python3 -m pip install -e .

# --- install upstream deps ---
echo "Installing upstream (robust install)..."

REQ1="openfold-3-mlx/requirements.txt"
REQ2="openfold-3-mlx/requirements-dev.txt"
PYPROJ="openfold-3-mlx/pyproject.toml"
SETUPPY="openfold-3-mlx/setup.py"

if [[ -f "$REQ1" ]]; then
  echo "Found requirements.txt"
  python3 -m pip install -r "$REQ1"
elif [[ -f "$REQ2" ]]; then
  echo "Found requirements-dev.txt"
  python3 -m pip install -r "$REQ2"
elif [[ -f "$PYPROJ" ]]; then
  echo "Found pyproject.toml → installing upstream editable"
  python3 -m pip install -e openfold-3-mlx
elif [[ -f "$SETUPPY" ]]; then
  echo "Found setup.py → installing upstream editable"
  python3 -m pip install -e openfold-3-mlx
else
  echo "WARNING: Could not find requirements.txt / pyproject.toml / setup.py in openfold-3-mlx."
  echo "You may need to follow upstream install instructions manually."
fi

echo ""
echo "Setup complete."
echo "Next: python -m macfold.cli fold --fasta examples/tiny.fasta --out results/run1"
