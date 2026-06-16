#!/usr/bin/env bash
# Launch the Qwen3-VL reranker service (FastAPI + sentence-transformers
# CrossEncoder, services/reranker/server.py). It runs out-of-process in its own
# conda env because the reranker needs transformers>=4.57 while the GME encoder
# pins 4.57-incompatible deps; the two cannot share one Python process. The
# query path / tuning experiments reach it over HTTP via KG4VD_RERANKER_URL.
#
# Defaults: Qwen3-VL-Reranker-2B (~5 GB), port 8003, GPU 1 (keeps GPU 0 free for
# the GME encoder). Endpoints: POST /score (one query) and POST /score_batch
# (many queries in one CrossEncoder pass - used by the batched experiments).
#
# Usage:
#   bash scripts/launch_reranker.sh
#   RERANKER_DEVICE=cuda:0 RERANKER_MODEL=Qwen/Qwen3-VL-Reranker-8B \
#       bash scripts/launch_reranker.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/paths.sh
source "$(dirname "$0")/paths.sh"   # RERANKER_ENV_PY, HF_HOME
ENV_PY="$RERANKER_ENV_PY"           # the reranker's conda env (services/reranker/README.md)
# In-repo server; override RERANKER_SERVER_PY only to relocate.
SERVER_PY="${RERANKER_SERVER_PY:-$REPO_ROOT/services/reranker/server.py}"
MODEL="${RERANKER_MODEL:-Qwen/Qwen3-VL-Reranker-2B}"
HOST="${RERANKER_HOST:-127.0.0.1}"
PORT="${RERANKER_PORT:-8003}"
DEVICE="${RERANKER_DEVICE:-cuda:1}"
DTYPE="${RERANKER_DTYPE:-bfloat16}"

if [[ ! -x "$ENV_PY" ]]; then
  echo "ERROR: reranker env python not found at $ENV_PY" >&2
  echo "       Create it per services/reranker/README.md," >&2
  echo "       or set RERANKER_ENV_PY to the right interpreter." >&2
  exit 1
fi
if [[ ! -f "$SERVER_PY" ]]; then
  echo "ERROR: reranker server.py not found at $SERVER_PY" >&2
  echo "       Set RERANKER_SERVER_PY to its location." >&2
  exit 1
fi

echo "Launching Qwen3-VL-Reranker: model=$MODEL host=$HOST port=$PORT device=$DEVICE dtype=$DTYPE"
exec "$ENV_PY" "$SERVER_PY" \
  --model "$MODEL" --host "$HOST" --port "$PORT" --device "$DEVICE" --dtype "$DTYPE"
