"""HTTP service for the Qwen3-VL multimodal reranker (sentence-transformers
CrossEncoder).

The reranker runs out-of-process because it needs transformers>=4.57, which is
incompatible in one Python process with the GME encoder's pinned stack. The
query path and tuning experiments reach it over HTTP via KG4VD_RERANKER_URL.
Launch it with ``scripts/launch_reranker.sh``.

Endpoints
---------
``POST /score`` - score one query against a list of documents::

  {
    "model":  "Qwen/Qwen3-VL-Reranker-2B",
    "text_1": "<query string>",
    "text_2": [
      "plain text doc",                            # (a) str
      "https://.../img.jpg",                       # (b) URL str (fetched as image)
      {"content": [                                # (c) OpenAI-style multimodal
        {"type": "text",      "text":      "..."},
        {"type": "image_url", "image_url": {"url": "file:///abs/path.png"}}
      ]}
    ],
    "instruction": "Retrieve text relevant to the user's query."   # optional
  }
  -> {"data": [{"index": i, "object": "score", "score": float}, ...], "model": "..."}

``POST /score_batch`` - score several queries in one CrossEncoder pass; fills
larger GPU batches and amortizes HTTP overhead for offline experiments::

  {"model": "...", "items": [{"text_1": "...", "text_2": [...]}, ...]}
  -> {"data": [{"index": item_i, "scores": [{"index": doc_i, "score": float}]}], ...}

``GET /health`` - liveness + loaded model/device.

Scores are sigmoid-activated to [0, 1] (the HF model-card recipe), giving a
comparable range across queries.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("kg4vd.reranker")


_DEFAULT_INSTRUCTION = "Retrieve text relevant to the user's query."


# ---------------------------------------------------------------------------
# Request shapes
# ---------------------------------------------------------------------------


class ScoreRequest(BaseModel):
    model: str | None = None
    text_1: str
    text_2: list[Any]
    instruction: str | None = None


class BatchScoreItem(BaseModel):
    text_1: str
    text_2: list[Any]


class BatchScoreRequest(BaseModel):
    model: str | None = None
    items: list[BatchScoreItem]
    instruction: str | None = None


# ---------------------------------------------------------------------------
# OpenAI-shape -> CrossEncoder-shape translation
# ---------------------------------------------------------------------------


def _doc_to_cross_encoder_input(doc: Any) -> str | dict[str, str]:
    """Translate one ``text_2[i]`` into a sentence-transformers doc.

    CrossEncoder accepts a ``str`` (text or image URL) or a mixed-modality
    ``{"text": ..., "image": ...}`` dict. We also accept the OpenAI-style
    ``{"content": [...]}`` shape the HTTP clients send and flatten it; ``file://``
    prefixes are stripped so PIL can open the local file directly.
    """
    if isinstance(doc, str):
        return doc

    if not isinstance(doc, dict):
        raise HTTPException(400, f"Unsupported document shape: {type(doc).__name__}")

    # OpenAI-style content array -> flatten to {text, image}.
    if "content" in doc and isinstance(doc["content"], list):
        text_parts: list[str] = []
        image_url: str | None = None
        for part in doc["content"]:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                if part.get("text"):
                    text_parts.append(str(part["text"]))
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url")
                if url and image_url is None:    # first image wins
                    image_url = url
        text = " ".join(text_parts).strip()
        out: dict[str, str] = {}
        if text:
            out["text"] = text
        if image_url:
            out["image"] = _normalise_image_uri(image_url)
        if not out:
            return ""
        return out if len(out) > 1 or "image" in out else out["text"]

    # Native CrossEncoder-style {text, image}.
    out = {}
    if doc.get("text"):
        out["text"] = str(doc["text"])
    if doc.get("image"):
        out["image"] = _normalise_image_uri(str(doc["image"]))
    if not out:
        return ""
    return out if len(out) > 1 or "image" in out else out["text"]


def _normalise_image_uri(uri: str) -> str:
    """``file://`` -> bare local path; everything else is passed through."""
    if uri.startswith("file://"):
        return os.path.abspath(uri[len("file://"):])
    return uri


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def build_app(model_name: str, *, device: str | None, dtype: str) -> FastAPI:
    logger.info("Loading CrossEncoder %s (device=%s, dtype=%s) …",
                model_name, device or "auto", dtype)
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    model = CrossEncoder(
        model_name, device=device, model_kwargs={"torch_dtype": torch_dtype}
    )
    logger.info("Model ready on device=%s", model.device)

    # Sigmoid so scores land in [0, 1] (HF model-card recipe) and stay
    # comparable across queries.
    sigmoid = torch.nn.Sigmoid()

    app = FastAPI(title="kg4vd Qwen3-VL reranker")

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": model_name, "device": str(model.device)}

    @app.post("/score")
    async def score(req: ScoreRequest):
        if not req.text_2:
            return {"data": [], "model": model_name}
        instruction = (req.instruction or _DEFAULT_INSTRUCTION).strip()
        pairs = [(req.text_1, _doc_to_cross_encoder_input(d)) for d in req.text_2]
        try:
            scores = model.predict(
                pairs, prompt=instruction, activation_fn=sigmoid, batch_size=64
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("predict() failed")
            raise HTTPException(500, f"predict() failed: {e!r}")
        return {
            "data": [
                {"index": i, "object": "score", "score": float(s)}
                for i, s in enumerate(scores)
            ],
            "model": model_name,
        }

    @app.post("/score_batch")
    async def score_batch(req: BatchScoreRequest):
        """Score multiple query-document lists in one CrossEncoder call -
        fills larger GPU batches and amortizes HTTP overhead. One instruction
        applies to the whole request."""
        if not req.items:
            return {"data": [], "model": model_name}
        instruction = (req.instruction or _DEFAULT_INSTRUCTION).strip()
        pairs: list[tuple[str, Any]] = []
        spans: list[tuple[int, int]] = []
        for item in req.items:
            start = len(pairs)
            for raw in item.text_2:
                pairs.append((item.text_1, _doc_to_cross_encoder_input(raw)))
            spans.append((start, len(pairs)))

        if pairs:
            try:
                scores = model.predict(
                    pairs, prompt=instruction, activation_fn=sigmoid, batch_size=64
                )
            except Exception as e:  # noqa: BLE001
                logger.exception("batch predict() failed")
                raise HTTPException(500, f"batch predict() failed: {e!r}")
            scores_list = [float(s) for s in scores]
        else:
            scores_list = []

        data = []
        for item_index, (start, end) in enumerate(spans):
            data.append({
                "index": item_index,
                "object": "score_batch_item",
                "scores": [
                    {"index": i - start, "object": "score", "score": scores_list[i]}
                    for i in range(start, end)
                ],
            })
        return {"data": data, "model": model_name}

    return app


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-VL-Reranker-2B")
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--device", default=None, help="cuda / cuda:0 / cpu (default: auto)")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()
    app = build_app(args.model, device=args.device, dtype=args.dtype)
    logger.info("Listening on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
