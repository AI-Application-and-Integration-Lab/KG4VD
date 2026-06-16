#!/usr/bin/env bash
# Build the MMKG for one or more recipes - construction only.
#
# Per recipe, the main build stages are:
#   ingest -> augment -> extract -> align -> cards -> embed -> index
#
# Optional stage:
#   retrieval_graph
#
# The pipeline is resumable. Re-running after an interruption will use
# --resume when <recipe_dir>/exp exists.
#
# ─────────────────────────────────────────────────────────────────
# Usage
# ─────────────────────────────────────────────────────────────────
#   bash scripts/build_mmkg.sh
#   bash scripts/build_mmkg.sh test
#   bash scripts/build_mmkg.sh recipes/foo/recipe.yaml recipes/bar/recipe.yaml
#   bash scripts/build_mmkg.sh --help
#
# ─────────────────────────────────────────────────────────────────
# Prereqs
# ─────────────────────────────────────────────────────────────────
#   1. conda env `kg4vd` active, or `kg4vd` available on PATH.
#      Example: pip install -e '.[gme]'
#   2. For LLM=remote, .env in repo root with OPENROUTER_API_KEY,
#      or OPENROUTER_API_KEY exported in the shell.
#   3. MinerU 3.x installed in its own env; MINERU_PYTHON points at that env's
#      python (see services/mineru/README.md).
#   4. A GPU for the GME encoder.
#
# ─────────────────────────────────────────────────────────────────
# Env knobs and defaults
# ─────────────────────────────────────────────────────────────────
#   FRESH=0                 If 1, rm -rf each recipe's exp/ before building.
#   SKIP_BUILD=0            If 1, skip the main 7-stage build.
#   SKIP_RETRIEVAL_GRAPH=1  If 1, skip retrieval_graph. Set to 0 to run it.
#   ENCODER_GPU=0           GPU pinned to GME encoder stages.
#   MINERU_PYTHON=...       Path to the MinerU 3.x env's python.
#   LLM=remote              Use recipe/OpenRouter remote LLM.
#   LLM=sglang              Use local sglang server. The script starts/stops
#                           sglang around GME stages to avoid GPU contention.
#   SGLANG_MODEL=...        Model name/path passed to launch_qwen36_sglang.sh.
#   SGLANG_URL=...          OpenAI-compatible local sglang endpoint.
#
# Outputs per recipe:
#   <recipe_dir>/exp/
#   <recipe_dir>/build.log
#   <recipe_dir>/build_rg.log

set -Eeuo pipefail
shopt -s nullglob

SCRIPT_NAME="$(basename "$0")"
ORIG_CWD="$PWD"
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$REPO_ROOT"

# shellcheck source=scripts/paths.sh
source "$REPO_ROOT/scripts/paths.sh"   # MINERU_PYTHON, HF_HOME, env names

# ─── Defaults ────────────────────────────────────────────────────
FRESH=${FRESH:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
SKIP_RETRIEVAL_GRAPH=${SKIP_RETRIEVAL_GRAPH:-1}
ENCODER_GPU=${ENCODER_GPU:-0}
LLM=${LLM:-remote}
SGLANG_MODEL=${SGLANG_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}
SGLANG_URL=${SGLANG_URL:-http://127.0.0.1:8004/v1}
SGLANG_STARTUP_TIMEOUT_SECONDS=${SGLANG_STARTUP_TIMEOUT_SECONDS:-900}
SGLANG_POLL_SECONDS=${SGLANG_POLL_SECONDS:-5}

# ─── Helpers ─────────────────────────────────────────────────────
usage () {
    cat <<EOF_USAGE
Usage:
  bash scripts/$SCRIPT_NAME [recipe_name|recipe.yaml ...]

Examples:
  bash scripts/$SCRIPT_NAME
  bash scripts/$SCRIPT_NAME test
  bash scripts/$SCRIPT_NAME recipes/foo/recipe.yaml recipes/bar/recipe.yaml

Environment:
  FRESH=1                 Delete each recipe's exp/ before building.
  SKIP_BUILD=1            Skip the main 7-stage build.
  SKIP_RETRIEVAL_GRAPH=0  Run the optional retrieval_graph stage.
  ENCODER_GPU=0           GPU used by GME encoder stages.
  LLM=remote              Use recipe/OpenRouter remote LLM. Default.
  LLM=sglang              Use local sglang server.
  MINERU_PYTHON=/path/python   MinerU 3.x env python.

sglang options:
  SGLANG_MODEL=$SGLANG_MODEL
  SGLANG_URL=$SGLANG_URL
EOF_USAGE
}

on_error () {
    local exit_code=$?
    local line_no=${BASH_LINENO[0]:-unknown}
    local cmd=${BASH_COMMAND:-unknown}
    echo >&2
    echo >&2 "ERROR: command failed with exit code $exit_code"
    echo >&2 "  line: $line_no"
    echo >&2 "  cmd : $cmd"
    exit "$exit_code"
}
trap on_error ERR

bool_is_0_or_1 () {
    local name="$1"
    local value="$2"
    if [[ "$value" != "0" && "$value" != "1" ]]; then
        echo "ERROR: $name must be 0 or 1, got '$value'." >&2
        exit 1
    fi
}

resolve_recipe_arg () {
    local a="$1"

    # Absolute or relative to repo root after cd.
    if [ -f "$a" ]; then
        realpath "$a"
        return 0
    fi

    # Relative to the caller's original cwd.
    if [ -f "$ORIG_CWD/$a" ]; then
        realpath "$ORIG_CWD/$a"
        return 0
    fi

    # Recipe name shorthand.
    if [ -f "$REPO_ROOT/recipes/$a/recipe.yaml" ]; then
        realpath "$REPO_ROOT/recipes/$a/recipe.yaml"
        return 0
    fi

    return 1
}

run_logged () {
    local log="$1"
    shift

    echo | tee -a "$log"
    printf '  [cmd]' | tee -a "$log"
    printf ' %q' "$@" | tee -a "$log"
    echo | tee -a "$log"

    "$@" 2>&1 | tee -a "$log"
}

# ─── CLI help ────────────────────────────────────────────────────
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

# ─── Validate env knobs ──────────────────────────────────────────
bool_is_0_or_1 FRESH "$FRESH"
bool_is_0_or_1 SKIP_BUILD "$SKIP_BUILD"
bool_is_0_or_1 SKIP_RETRIEVAL_GRAPH "$SKIP_RETRIEVAL_GRAPH"

case "$LLM" in
    remote|sglang)
        ;;
    "")
        LLM=remote
        ;;
    *)
        echo "ERROR: unsupported LLM='$LLM'. Use LLM=remote or LLM=sglang." >&2
        exit 1
        ;;
esac

# ─── 1. Resolve recipe list ──────────────────────────────────────
RECIPES=()

if [ $# -gt 0 ]; then
    for a in "$@"; do
        if recipe_path="$(resolve_recipe_arg "$a")"; then
            RECIPES+=("$recipe_path")
        else
            echo "  ⚠️  no recipe for '$a' - tried direct path, cwd-relative path, and recipes/$a/recipe.yaml; skipping"
        fi
    done
else
    for r in "$REPO_ROOT"/recipes/*/recipe.yaml; do
        [ -f "$r" ] && RECIPES+=("$(realpath "$r")")
    done
fi

if [ ${#RECIPES[@]} -eq 0 ]; then
    echo "ERROR: no recipes to build." >&2
    exit 1
fi

# Deduplicate while preserving order.
declare -A SEEN_RECIPES=()
DEDUPED_RECIPES=()
for r in "${RECIPES[@]}"; do
    if [[ -z "${SEEN_RECIPES[$r]:-}" ]]; then
        SEEN_RECIPES[$r]=1
        DEDUPED_RECIPES+=("$r")
    fi
done
RECIPES=("${DEDUPED_RECIPES[@]}")

# ─── 2. CLI sanity check ─────────────────────────────────────────
if ! command -v kg4vd >/dev/null 2>&1; then
    echo "ERROR: 'kg4vd' not found on PATH." >&2
    echo "  Try: conda activate kg4vd" >&2
    echo "  Then: pip install -e '.[gme]'" >&2
    exit 1
fi

# ─── 3. .env ─────────────────────────────────────────────────────
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "$REPO_ROOT/.env"
    set +a
else
    echo "WARN: no .env at $REPO_ROOT/.env - assuming needed env vars are already exported"
fi

# ─── 3b. LLM selection ───────────────────────────────────────────
LLM_OVERRIDES=()

if [ "$LLM" = "sglang" ]; then
    echo "[build_mmkg] LLM=sglang → $SGLANG_URL ($SGLANG_MODEL)"
    LLM_OVERRIDES=(
        --set "generator.llm.kind=sglang"
        --set "generator.llm.model=$SGLANG_MODEL"
        --set "generator.llm.base_url=$SGLANG_URL"
        --set "generator.llm.api_key_env="
    )
else
    : "${OPENROUTER_API_KEY:?ERROR: OPENROUTER_API_KEY not set; put it in .env, export it, or use LLM=sglang.}"
fi

# ─── 4. MinerU env python ────────────────────────────────────────
# MinerU 3.x runs out-of-process in its own env; ingest shells out to
# services/mineru/run.py with this python (see services/mineru/README.md).
export MINERU_PYTHON
if [ ! -x "$MINERU_PYTHON" ]; then
    echo "ERROR: MinerU env python not found at $MINERU_PYTHON." >&2
    echo "  Set MINERU_PYTHON=/path/to/mineru-env/bin/python and re-run." >&2
    exit 1
fi

# ─── 5. GPU for the encoder ──────────────────────────────────────
# Do not export CUDA_VISIBLE_DEVICES globally. In LLM=sglang mode, sglang may
# need all GPUs, while GME should be pinned only during GME stages.
GME_ENV=(env "CUDA_VISIBLE_DEVICES=$ENCODER_GPU")

# ─── 5b. sglang lifecycle, only used for LLM=sglang ──────────────
SGLANG_PGID=""
SGLANG_STARTED_BY_SCRIPT=0

sglang_is_ready () {
    curl -sf "$SGLANG_URL/models" >/dev/null 2>&1
}

sglang_up () {
    if sglang_is_ready; then
        echo "  [build_mmkg] reusing existing sglang at $SGLANG_URL"
        SGLANG_STARTED_BY_SCRIPT=0
        return 0
    fi

    echo "  [build_mmkg] starting sglang ($SGLANG_MODEL) …"
    QWEN36_MODEL_PATH="$SGLANG_MODEL" setsid bash scripts/launch_qwen36_sglang.sh \
        >"$REPO_ROOT/sglang.log" 2>&1 &

    SGLANG_PGID=$!
    SGLANG_STARTED_BY_SCRIPT=1

    local elapsed=0
    while [ "$elapsed" -lt "$SGLANG_STARTUP_TIMEOUT_SECONDS" ]; do
        if sglang_is_ready; then
            echo "  [build_mmkg] sglang ready"
            return 0
        fi

        if ! kill -0 "$SGLANG_PGID" 2>/dev/null; then
            echo "ERROR: sglang exited during startup. Last log lines:" >&2
            tail -40 "$REPO_ROOT/sglang.log" >&2 || true
            exit 1
        fi

        sleep "$SGLANG_POLL_SECONDS"
        elapsed=$((elapsed + SGLANG_POLL_SECONDS))
    done

    echo "ERROR: sglang not ready after ${SGLANG_STARTUP_TIMEOUT_SECONDS}s. Last log lines:" >&2
    tail -40 "$REPO_ROOT/sglang.log" >&2 || true
    exit 1
}

sglang_down () {
    if [ "$SGLANG_STARTED_BY_SCRIPT" != "1" ]; then
        return 0
    fi

    if [ -z "$SGLANG_PGID" ]; then
        return 0
    fi

    echo "  [build_mmkg] stopping sglang …"
    kill -TERM -"$SGLANG_PGID" 2>/dev/null || true
    wait "$SGLANG_PGID" 2>/dev/null || true

    SGLANG_PGID=""
    SGLANG_STARTED_BY_SCRIPT=0

    # Give CUDA/VRAM a moment to settle before GME loads.
    sleep 3
}

cleanup () {
    set +e
    sglang_down
}
trap cleanup EXIT

# ─── 6. Summary ──────────────────────────────────────────────────
echo
echo "=================================================================="
echo "  KG4VD MMKG construction"
echo "  repo root    = $REPO_ROOT"
echo "  encoder GPU  = $ENCODER_GPU"
echo "  LLM          = $LLM"
echo "  MinerU py    = $MINERU_PYTHON"
echo "  fresh        = $FRESH"
echo "  skip build   = $SKIP_BUILD"
echo "  skip rg      = $SKIP_RETRIEVAL_GRAPH"
echo "  recipes      ="
for r in "${RECIPES[@]}"; do
    echo "    - $r"
done
echo "=================================================================="

# ─── 7. Build logic ──────────────────────────────────────────────
build_one () {
    local recipe="$1"
    local recipe_dir
    recipe_dir="$( dirname "$recipe" )"

    if ! compgen -G "$recipe_dir/data/*.pdf" > /dev/null; then
        echo "  ⏭  $recipe: no PDFs in $recipe_dir/data/ - skipping"
        return 0
    fi

    echo
    echo "------------------------------------------------------------------"
    echo "  $recipe"
    echo "------------------------------------------------------------------"

    if [ "$FRESH" = "1" ]; then
        echo "  [build_mmkg] FRESH=1 - deleting $recipe_dir/exp/"
        rm -rf -- "$recipe_dir/exp"
    fi

    local log="$recipe_dir/build.log"
    local rg_log="$recipe_dir/build_rg.log"
    local resume_args=()

    if [ -d "$recipe_dir/exp" ]; then
        resume_args=(--resume)
    fi

    if [ "$SKIP_BUILD" = "1" ]; then
        echo "  [build_mmkg] SKIP_BUILD=1 - skipping main build"
    elif [ "$LLM" = "sglang" ]; then
        : > "$log"
        echo "  [build_mmkg] staged build with LLM=sglang → $log"

        # Staged to avoid GME and local LLM co-residing on GPUs:
        #   GME-free: ingest
        #   sglang : augment, extract
        #   GME    : align-embed
        #   sglang : align judge, using node_embs.npz
        #   GME    : cards, embed, index
        run_logged "$log" "${GME_ENV[@]}" kg4vd build "$recipe" --stages ingest "${resume_args[@]}"

        sglang_up
        run_logged "$log" kg4vd build "$recipe" --stages augment,extract --resume "${LLM_OVERRIDES[@]}"
        sglang_down

        run_logged "$log" "${GME_ENV[@]}" kg4vd align-embed "$recipe"

        sglang_up
        run_logged "$log" kg4vd build "$recipe" --stages align --resume "${LLM_OVERRIDES[@]}"
        sglang_down

        run_logged "$log" "${GME_ENV[@]}" kg4vd build "$recipe" --stages cards,embed,index --resume
    else
        : > "$log"
        echo "  [build_mmkg] straight build with remote LLM ${resume_args[*]:-} → $log"

        # Remote LLM: local GPU is used by GME only; LLM stages call API.
        run_logged "$log" "${GME_ENV[@]}" kg4vd build "$recipe" "${resume_args[@]}" "${LLM_OVERRIDES[@]}"
    fi

    if [ "$SKIP_RETRIEVAL_GRAPH" != "1" ]; then
        : > "$rg_log"
        echo "  [build_mmkg] retrieval_graph → $rg_log"
        run_logged "$rg_log" kg4vd build "$recipe" --stages retrieval_graph --resume
    else
        echo "  [build_mmkg] SKIP_RETRIEVAL_GRAPH=1 - skipping retrieval_graph"
    fi

    echo "  [build_mmkg] DONE $recipe"
}

for r in "${RECIPES[@]}"; do
    build_one "$r"
done

echo
echo "=================================================================="
echo "  All requested MMKGs built."
echo "  Artefacts under <recipe_dir>/exp/"
echo "  Logs at <recipe_dir>/build.log and <recipe_dir>/build_rg.log"
echo "=================================================================="
