#!/bin/bash
# SimpleLLM setup
# Requires: Python 3.12, NVIDIA GPU with CUDA 12.8+

set -e

MODEL_DIR="${1:-./gpt-oss-120b}"
VENV_DIR="${2:-./.venv}"

echo "=== SimpleLLM setup ==="

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create and activate venv
uv venv --python 3.12 "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Install dependencies
echo "Installing PyTorch 2.9.1..."
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

echo "Installing flash-attn 2.8.3..."
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

echo "Installing other dependencies..."
uv pip install numpy safetensors tokenizers huggingface_hub einops triton tqdm

# Download model
echo "Downloading model..."
hf download openai/gpt-oss-120b --local-dir "$MODEL_DIR"

echo ""
echo "Done! Run:"
echo "  source $VENV_DIR/bin/activate"
echo "You are good to go! Explore the examples in the /cookbook directory."