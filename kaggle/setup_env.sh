#!/usr/bin/env bash
# Builds the pinned Python 3.12 environment at $VENV. Kaggle's image runs
# Python 3.13, which numpy 1.26 and fairseq2 (omnilingual-asr) do not support,
# so we never install into the system interpreter.
# Usage: bash setup_env.sh [--omnilingual]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-/tmp/waxal-venv}"

nvidia-smi --query-gpu=name,memory.total --format=csv || echo "No GPU detected"

if [ ! -x "$VENV/bin/python" ]; then
  pip install --quiet uv
  uv venv --python 3.12 "$VENV"  # uv downloads a managed 3.12 if none is installed
fi
uv pip install --quiet --python "$VENV/bin/python" -r "$REPO/requirements.txt"
if [ "${1:-}" = "--omnilingual" ]; then
  uv pip install --quiet --python "$VENV/bin/python" -r "$REPO/requirements-omnilingual.txt"
fi

"$VENV/bin/python" -c "import torch, transformers, datasets; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.device_count(), '| transformers', transformers.__version__, '| datasets', datasets.__version__)"
df -h /tmp /kaggle/working 2>/dev/null || true
