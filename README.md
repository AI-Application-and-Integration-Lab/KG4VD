# KG4VD

Code and dataset release for **Multimodal Graph RAG for Long-range Visually Rich
Document Understanding**.

KG4VD is a multimodal graph-based retrieval-augmented generation framework for
long, visually rich documents. It builds a multimodal knowledge graph (MMKG)
from document pages, indexes graph-grounded evidence, and answers questions by
combining page retrieval with graph-based evidence propagation.

The repository also releases **DLVQA**, a document-level visual question
answering benchmark designed for global, holistic document understanding.

## Installation

KG4VD uses separate environments because MinerU, GME-Qwen2-VL, the reranker,
and optional local LLM serving stacks require incompatible dependencies.

```bash
bash scripts/setup_envs.sh
cp .env.example .env
conda activate kg4vd
kg4vd --help
```

See [INSTALL.md](INSTALL.md) for the full environment matrix and manual setup.

KG4VD does not require sglang. The LLM backend can be OpenRouter, OpenAI, or a
local OpenAI-compatible server such as sglang. OpenRouter is the default setup
used by the provided recipes; sglang is only needed if you want to run a local
LLM server.

## Quick Start

Build the test recipe:

```bash
conda activate kg4vd
kg4vd build recipes/test/recipe.yaml --resume
```

Query a built index:

```bash
kg4vd query recipes/test/recipe.yaml -q "Your question"
kg4vd query-batch recipes/test/recipe.yaml -i recipes/test/questions.jsonl
```

Inspect a run:

```bash
kg4vd show-config recipes/test/recipe.yaml
kg4vd report recipes/test/recipe.yaml --kind per_stage
python scripts/inspect_run.py recipes/test/exp --page 1
```

For scripted builds:

```bash
bash scripts/build_mmkg.sh test
LLM=sglang bash scripts/build_mmkg.sh test   # optional local LLM
```

## DLVQA

DLVQA is a document-level VQA benchmark for questions that require global
document comprehension. Each example includes answer guidance, a reference
summary, and supporting facts.

The released benchmark contains 525 questions over four long documents. See
[datasets/dlvqa/README.md](datasets/dlvqa/README.md) for the schema and
document-level details.

## Development

```bash
conda activate kg4vd
ruff check src scripts services
python -m pytest -q
```

Some tests and workflows require the optional service environments described in
[INSTALL.md](INSTALL.md).

## License

See [LICENSE](LICENSE).
