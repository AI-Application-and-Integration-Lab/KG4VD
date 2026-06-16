# Shared machine paths + conda env names for kg4vd scripts.
#
# Sourced by the other scripts (launch_*.sh, build_mmkg.sh, setup_envs.sh) so
# box-specific paths live in ONE place. Every value is overridable from the
# environment. CONDA_BASE is derived from `conda info --base` when conda is on
# PATH, else it falls back to the path below - edit that one line per machine.

_DEFAULT_CONDA_BASE="/media/ai2lab/storage1/miniconda3"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$_DEFAULT_CONDA_BASE")}"
CONDA_SH="${CONDA_SH:-$CONDA_BASE/etc/profile.d/conda.sh}"

# Conda env names. Override from the shell if you use different names.
KG4VD_ENV="${KG4VD_ENV:-kg4vd}"
MINERU_ENV="${MINERU_ENV:-mineru}"
RERANKER_ENV="${RERANKER_ENV:-kg4vd_reranker}"
SGLANG_ENV="${SGLANG_ENV:-kg4vd_sglang}"

# Derived interpreters (the out-of-process services run in their own envs).
MINERU_PYTHON="${MINERU_PYTHON:-$CONDA_BASE/envs/$MINERU_ENV/bin/python}"
RERANKER_ENV_PY="${RERANKER_ENV_PY:-$CONDA_BASE/envs/$RERANKER_ENV/bin/python}"

# HuggingFace model cache (shared by GME / MinerU / reranker / sglang).
export HF_HOME="${HF_HOME:-/media/ai2lab/storage1/hf_cache}"
