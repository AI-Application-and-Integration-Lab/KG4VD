#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/paths.sh
source "$(dirname "$0")/paths.sh"   # CONDA_SH, SGLANG_ENV, HF_HOME

CONDA_ENV="${CONDA_ENV:-$SGLANG_ENV}"
MODEL_PATH="${QWEN36_MODEL_PATH:-Qwen/Qwen3.6-35B-A3B-FP8}"
HOST="${QWEN36_HOST:-127.0.0.1}"
PORT="${QWEN36_PORT:-8004}"
TP_SIZE="${QWEN36_TP_SIZE:-2}"
CONTEXT_LENGTH="${QWEN36_CONTEXT_LENGTH:-32768}"
MEM_FRACTION="${QWEN36_MEM_FRACTION:-0.82}"
# HF cache_dir. Must point at the *hub* dir (HF_HOME/hub) where snapshots
# actually live as models--<org>--<name>/snapshots/<rev>; pointing at the
# HF_HOME root makes sglang miss the cached snapshot and re-download.
DOWNLOAD_DIR="${QWEN36_DOWNLOAD_DIR:-$HF_HOME/hub}"
LOG_LEVEL="${QWEN36_LOG_LEVEL:-info}"
# Blackwell (RTX 5090, sm_120) + CUDA 13: the default flashinfer attention
# backend hangs at startup ("using attn output gate!", server never serves).
# Override when another attention backend is preferred.
ATTN_BACKEND="${QWEN36_ATTN_BACKEND:-triton}"
# flashinfer's sampling kernel JIT-compiles at first inference and fails to
# link (`ld: cannot find -lcuda` under the conda toolchain), crashing the
# server on the first request. The pytorch sampling backend needs no JIT.
SAMPLING_BACKEND="${QWEN36_SAMPLING_BACKEND:-pytorch}"

# conda's CUDA activation scripts reference unset vars (e.g.
# NVCC_PREPEND_FLAGS) - relax `set -u` just for the activate step so they
# don't abort the launch under a non-interactive shell.
set +u
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u

# conda's activate scripts export HOST=<build-triplet> (e.g.
# x86_64-conda-linux-gnu), clobbering our bind address. Re-assert it
# (and PORT, defensively) AFTER activation so --host stays 127.0.0.1.
HOST="${QWEN36_HOST:-127.0.0.1}"
PORT="${QWEN36_PORT:-8004}"

export HF_HOME                       # already set by paths.sh; survive conda activate
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --context-length "${CONTEXT_LENGTH}" \
  --mem-fraction-static "${MEM_FRACTION}" \
  --download-dir "${DOWNLOAD_DIR}" \
  --attention-backend "${ATTN_BACKEND}" \
  --sampling-backend "${SAMPLING_BACKEND}" \
  --enable-multimodal \
  --reasoning-parser qwen3 \
  --log-level "${LOG_LEVEL}"
