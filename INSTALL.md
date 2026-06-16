# Installing kg4vd

kg4vd spans multiple environments because heavyweight model stacks have
mutually incompatible `transformers` pins and can't share one Python process:

| Env | Purpose | Key constraint | Used by |
|---|---|---|---|
| **`kg4vd`** (main) | core pipeline + GME-Qwen2-VL encoder | `transformers<4.52` (GME remote code) | everything |
| **`mineru`** | MinerU 3.x PDF ingest | `transformers>=4.57.3` | `build` (ingest stage) |
| **`kg4vd_reranker`** | Qwen3-VL reranker HTTP service | `transformers>=4.57` | `query` (optional) |
| **your sglang env** | local OpenAI-compatible LLM server | model-dependent | `build`/`query` with a local LLM (optional) |

The model envs run out-of-process: ingest shells out to MinerU
(`MINERU_PYTHON`), the reranker is reached over HTTP (`KG4VD_RERANKER_URL`),
and a local LLM is reached through an OpenAI-compatible endpoint.

## Prerequisites

- Conda / Miniconda.
- An NVIDIA GPU + recent CUDA. On Blackwell (RTX 5090, sm_120) install torch
  from the **cu128+** wheel index (cu126 lacks sm_120 kernels).

## Quick start (automated)

Creates the three managed conda envs and installs each:

```bash
bash scripts/setup_envs.sh                 # all three; sglang -> see below
# or one at a time:
bash scripts/setup_envs.sh kg4vd
bash scripts/setup_envs.sh mineru
bash scripts/setup_envs.sh reranker
```

Knobs (env vars): `CONDA_BASE`, `KG4VD_ENV` / `MINERU_ENV` / `RERANKER_ENV`
(env names), `TORCH_INDEX` (default `https://download.pytorch.org/whl/cu128`),
`PYVER` (default `3.10`).

Then configure and verify the CLI:

```bash
cp .env.example .env        # fill OPENROUTER_API_KEY, MINERU_PYTHON, ...
conda run -n kg4vd kg4vd --help
```

## Manual install (per env)

### 1. `kg4vd` - core + GME encoder
```bash
conda create -n kg4vd python=3.10 -y
conda run -n kg4vd pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
conda run -n kg4vd pip install -e '.[gme,viz,dev]'
```

### 2. `mineru` - MinerU 3.x ingest
```bash
conda create -n mineru python=3.10 -y
conda run -n mineru pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
conda run -n mineru pip install -r services/mineru/requirements.txt
```
See [`services/mineru/README.md`](./services/mineru/README.md). Point ingest at
this env with `MINERU_PYTHON=<conda>/envs/mineru/bin/python`.

### 3. `kg4vd_reranker` - reranker service (optional, query only)
```bash
conda create -n kg4vd_reranker python=3.10 -y
conda run -n kg4vd_reranker pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
conda run -n kg4vd_reranker pip install -r services/reranker/requirements.txt
```
See [`services/reranker/README.md`](./services/reranker/README.md). Launch with
`bash scripts/launch_reranker.sh`.

### 4. sglang LLM server (optional - only for a local LLM)
Not scripted here - follow the official docs:
**https://docs.sglang.ai/** (install into its own env). kg4vd expects an
OpenAI-compatible server. `scripts/launch_qwen36_sglang.sh` is an example
launcher for the model used in our experiments. If you use a remote API
(OpenRouter/OpenAI), you do not need sglang.

## Environment variables (`.env`)

`cp .env.example .env`, then set:
- `OPENROUTER_API_KEY` - LLM provider for extract/align/augment/generation (unless `LLM=sglang`).
- `MINERU_PYTHON` - path to the MinerU env's python (ingest).
- `KG4VD_RERANKER_URL` - optional; defaults to `http://127.0.0.1:8003`.

## Which envs do I actually need?

| Task | Needs |
|---|---|
| `kg4vd build` (OpenRouter LLM) | `kg4vd` + `mineru` + `OPENROUTER_API_KEY` |
| `kg4vd build` (`LLM=sglang`) | `kg4vd` + `mineru` + sglang |
| `kg4vd query` (QGGE, OpenRouter) | `kg4vd` + `OPENROUTER_API_KEY` |
| `kg4vd query` (reranker on / local LLM) | `kg4vd` + `kg4vd_reranker` and/or sglang |

The query path auto-starts/stops the reranker + sglang servers (`--no-manage-*`
to run them yourself).
