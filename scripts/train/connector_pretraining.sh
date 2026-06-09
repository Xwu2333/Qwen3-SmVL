#!/bin/bash
# Launch Qwen3-SmVL connector pretraining.
# Mirrors the env-setup / debug-print style of smolvlm2's multinode26-256M.sh.

set -x -e

# ---------------------------------------------------------------------------
# Resolve project root from this script's own location, so the script works
# no matter which directory you invoke it from.
#   scripts/train/connector_pretraining.sh  ->  project root is ../..
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Hugging Face cache directory (kept inside the project to avoid polluting $HOME)
# ---------------------------------------------------------------------------
export HF_HOME="${PROJECT_ROOT}/.cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# ---------------------------------------------------------------------------
# Detect platform and activate the venv virtual environment accordingly
#   - Linux / macOS / WSL : <venv>/bin/activate
#   - Windows (Git Bash / MSYS / Cygwin) : <venv>/Scripts/activate
# ---------------------------------------------------------------------------
VENV_DIR="${PROJECT_ROOT}/.venv"

case "$(uname -s)" in
    Linux*|Darwin*)
        PLATFORM="unix"
        ACTIVATE_SCRIPT="${VENV_DIR}/bin/activate"
        ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
        PLATFORM="windows"
        ACTIVATE_SCRIPT="${VENV_DIR}/Scripts/activate"
        ;;
    *)
        echo "Unknown platform: $(uname -s)" >&2
        exit 1
        ;;
esac

echo "Detected platform: ${PLATFORM}"
echo "Activating venv: ${ACTIVATE_SCRIPT}"
# shellcheck disable=SC1090
source "${ACTIVATE_SCRIPT}"

# ---------------------------------------------------------------------------
# Debug prints
# ---------------------------------------------------------------------------
echo "Python path: $(which python)"
python -c "import sys; print('Sys path:', sys.path)"
python -c "import torch; print('PyTorch version:', torch.__version__, '\nPyTorch location:', torch.__file__)"
python -c "import transformers; print('Transformers version:', transformers.__version__, '\nTransformers location:', transformers.__file__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| device count:', torch.cuda.device_count())"
python -c "import swanlab; print('SwanLab version:', swanlab.__version__, '\nSwanLab location:', swanlab.__file__)"
which accelerate
which swanlab

# ---------------------------------------------------------------------------
# User-defined variables
# ---------------------------------------------------------------------------
CONFIG_YAML="scripts/train/connector_pretraining.yaml"
NUM_PROCESSES=${NUM_PROCESSES:-4}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29500}

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "CONFIG_YAML=$CONFIG_YAML"
echo "NUM_PROCESSES=$NUM_PROCESSES"
echo "MAIN_PROCESS_PORT=$MAIN_PROCESS_PORT"

# ---------------------------------------------------------------------------
# Launch training
# ---------------------------------------------------------------------------
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

if [ "${NUM_PROCESSES}" -gt 1 ]; then
    accelerate launch \
        --num_processes="${NUM_PROCESSES}" \
        --main_process_port="${MAIN_PROCESS_PORT}" \
        -m qwen3smvl.train.train \
        "${CONFIG_YAML}"
else
    python -m qwen3smvl.train.train "${CONFIG_YAML}"
fi
