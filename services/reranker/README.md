# services/reranker

HTTP service wrapping **Qwen3-VL-Reranker** (a multimodal `sentence_transformers.CrossEncoder`)
for reranking retrieved page candidates against a query.

It runs **out-of-process** in its own conda env: the reranker needs
`transformers>=4.57`, incompatible in one Python process with the GME encoder's
pinned stack. The query path / `tuning/experiments` reach it over HTTP via
`KG4VD_RERANKER_URL`.

## Setup (one-time)

```bash
conda create -n kg4vd_reranker python=3.10 -y
conda activate kg4vd_reranker
pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128   # cu128: RTX 5090 / sm_120
pip install -r services/reranker/requirements.txt
```

The 2B variant loads ~5 GB (cached under `$HF_HOME/hub`); 8B is ~16 GB.

## Run

```bash
bash scripts/launch_reranker.sh                 # 2B, 127.0.0.1:8003, GPU 1
RERANKER_DEVICE=cuda:0 RERANKER_MODEL=Qwen/Qwen3-VL-Reranker-8B \
    bash scripts/launch_reranker.sh
```

Override via env: `RERANKER_MODEL`, `RERANKER_PORT`, `RERANKER_DEVICE`,
`RERANKER_DTYPE`, `RERANKER_HOST`, `RERANKER_ENV_PY`.

## Endpoints

- `POST /score` - one query vs N docs → `{"data": [{"index", "score"}], "model"}`.
- `POST /score_batch` - many queries in one CrossEncoder pass (fills the GPU
  batch; used by the batched experiments) → `{"data": [{"index", "scores": [...]}]}`.
- `GET /health` - liveness + loaded model/device.

Docs (`text_2`) may be plain strings, image URLs, or OpenAI-style
`{"content": [{"type": "text", ...}, {"type": "image_url", "image_url": {"url": "file:///..."}}]}`.
Scores are sigmoid-activated to `[0, 1]`.

## Parity check

```bash
conda run -n kg4vd_reranker python services/reranker/test_parity.py --mode direct
conda run -n kg4vd_reranker python services/reranker/test_parity.py --mode http
```

Asserts the model-card reference scores (±0.05) and the published ranking.
