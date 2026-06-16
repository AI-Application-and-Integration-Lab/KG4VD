"""Parity test against the Qwen3-VL-Reranker-2B model-card example.

The HF model card (https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) publishes
the exact sigmoid-activated scores for this 1-query / 3-doc configuration::

    query    = "A woman playing with her dog on a beach at sunset."
    docs     = [text, image_url, {text+image}]
    prompt   = "Retrieve images or text relevant to the user's query."
    expected = [0.8594, 0.6367, 0.7891]       # sigmoid-activated

Numerics drift a little across GPUs, so we use a loose tolerance (±0.05) plus a
stronger ordering invariant (score[0] > score[2] > score[1]).

Run modes:
  # in-process CrossEncoder
  python services/reranker/test_parity.py --mode direct
  # HTTP against a running server (scripts/launch_reranker.sh)
  python services/reranker/test_parity.py --mode http --url http://127.0.0.1:8003

Exits 0 on parity, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys

QUERY = "A woman playing with her dog on a beach at sunset."
DOC_TEXT = (
    "A woman shares a joyful moment with her golden retriever on a "
    "sun-drenched beach at sunset, as the dog offers its paw in a "
    "heartwarming display of companionship and trust."
)
IMG_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
PROMPT = "Retrieve images or text relevant to the user's query."

EXPECTED = [0.8594, 0.6367, 0.7891]
TOLERANCE = 0.05


def _check(scores: list[float]) -> bool:
    print(f"  got:      {[round(s, 4) for s in scores]}")
    print(f"  expected: {EXPECTED}")
    print(f"  diff:     {[round(s - e, 4) for s, e in zip(scores, EXPECTED)]}")
    abs_ok = all(abs(s - e) <= TOLERANCE for s, e in zip(scores, EXPECTED))
    order_ok = scores[0] > scores[2] > scores[1]  # text-only > text+image > image-only
    if abs_ok and order_ok:
        print("  PASS - within tolerance and ranking matches the model card.")
        return True
    print("  FAIL -", "scores drifted >tol" if not abs_ok else "",
          "ranking mismatch" if not order_ok else "")
    return False


def _run_direct() -> bool:
    import torch
    from sentence_transformers import CrossEncoder

    print("Loading CrossEncoder(Qwen/Qwen3-VL-Reranker-2B)…")
    model = CrossEncoder(
        "Qwen/Qwen3-VL-Reranker-2B", model_kwargs={"torch_dtype": torch.bfloat16}
    )
    docs = [DOC_TEXT, IMG_URL, {"text": DOC_TEXT, "image": IMG_URL}]
    pairs = [(QUERY, d) for d in docs]
    print("Running predict() with sigmoid activation…")
    scores = model.predict(pairs, prompt=PROMPT, activation_fn=torch.nn.Sigmoid())
    return _check([float(s) for s in scores])


def _run_http(url: str) -> bool:
    import httpx

    body = {
        "model": "Qwen/Qwen3-VL-Reranker-2B",
        "text_1": QUERY,
        "text_2": [
            DOC_TEXT,
            IMG_URL,
            {"content": [
                {"type": "text", "text": DOC_TEXT},
                {"type": "image_url", "image_url": {"url": IMG_URL}},
            ]},
        ],
        "instruction": PROMPT,
    }
    print(f"POST {url}/score …")
    r = httpx.post(f"{url.rstrip('/')}/score", json=body, timeout=120.0)
    r.raise_for_status()
    payload = r.json()
    print(f"  raw response: {json.dumps(payload, indent=2)[:400]}…")
    return _check([item["score"] for item in payload["data"]])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["direct", "http"], default="direct")
    p.add_argument("--url", default="http://127.0.0.1:8003")
    args = p.parse_args()
    print(f"Mode: {args.mode}")
    ok = _run_direct() if args.mode == "direct" else _run_http(args.url)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
