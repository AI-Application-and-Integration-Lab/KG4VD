#!/usr/bin/env bash
# Create + install the kg4vd conda environments. See INSTALL.md for the full
# story. Three conda envs are handled here; the sglang LLM server has its own
# install (https://docs.sglang.ai/) and is only needed for a local LLM.
#
#   bash scripts/setup_envs.sh                 # all managed envs
#   bash scripts/setup_envs.sh kg4vd           # just one: kg4vd | mineru | reranker
#
# Knobs (env vars):
#   CONDA_BASE   conda install root (default: scripts/paths.sh / `conda info --base`)
#   KG4VD_ENV / MINERU_ENV / RERANKER_ENV    env names
#   TORCH_INDEX  torch wheel index matching your CUDA/driver setup
#   PYVER        python version for new envs (default 3.10)
#   RECREATE=1   conda env remove + recreate even if it exists
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/paths.sh
source "$(dirname "$0")/paths.sh"   # CONDA_BASE + env names (KG4VD_ENV/MINERU_ENV/RERANKER_ENV)
CONDA="$CONDA_BASE/bin/conda"
# Default matches the setup used in our experiments. Override for your CUDA stack.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
PYVER="${PYVER:-3.10}"

command -v "$CONDA" >/dev/null 2>&1 || { echo "ERROR: conda not found at $CONDA (set CONDA_BASE)." >&2; exit 1; }

ensure_env () {  # ensure_env <name>
    local name="$1"
    if "$CONDA" env list | awk '{print $1}' | grep -qx "$name"; then
        if [ "${RECREATE:-0}" = "1" ]; then
            echo "[setup] recreating env '$name'"; "$CONDA" env remove -n "$name" -y
        else
            echo "[setup] env '$name' exists - reusing (RECREATE=1 to rebuild)"; return 0
        fi
    fi
    echo "[setup] creating env '$name' (python $PYVER)"
    "$CONDA" create -n "$name" "python=$PYVER" -y
}

pip_in () { "$CONDA" run -n "$1" pip "${@:2}"; }

setup_kg4vd () {
    echo "=== kg4vd (core + GME encoder) ==="
    ensure_env "$KG4VD_ENV"
    pip_in "$KG4VD_ENV" install torch torchvision --index-url "$TORCH_INDEX"
    pip_in "$KG4VD_ENV" install -e '.[gme,viz,dev]'
    echo "[setup] kg4vd done. Smoke: conda run -n $KG4VD_ENV kg4vd --help"
}

setup_mineru () {
    echo "=== mineru (MinerU 3.x ingest) ==="
    ensure_env "$MINERU_ENV"
    pip_in "$MINERU_ENV" install torch torchvision --index-url "$TORCH_INDEX"
    pip_in "$MINERU_ENV" install -r services/mineru/requirements.txt
    echo "[setup] mineru done. Point ingest at it: MINERU_PYTHON=$CONDA_BASE/envs/$MINERU_ENV/bin/python"
}

setup_reranker () {
    echo "=== reranker (Qwen3-VL reranker service) ==="
    ensure_env "$RERANKER_ENV"
    pip_in "$RERANKER_ENV" install torch==2.8.0 torchvision==0.23.0 --index-url "$TORCH_INDEX"
    pip_in "$RERANKER_ENV" install -r services/reranker/requirements.txt
    echo "[setup] reranker done. Launch: bash scripts/launch_reranker.sh"
}

target="${1:-all}"
case "$target" in
    kg4vd)    setup_kg4vd ;;
    mineru)   setup_mineru ;;
    reranker) setup_reranker ;;
    all)      setup_kg4vd; setup_mineru; setup_reranker ;;
    *) echo "usage: $0 [all|kg4vd|mineru|reranker]" >&2; exit 2 ;;
esac

cat <<EOF

Done ($target). Next:
  cp .env.example .env        # fill OPENROUTER_API_KEY, MINERU_PYTHON
  conda run -n $KG4VD_ENV python -m pytest -q

The sglang LLM server (optional, local LLM) installs separately - see
https://docs.sglang.ai/ and scripts/launch_qwen36_sglang.sh. With a remote API
(OpenRouter/OpenAI) you don't need it.
EOF
