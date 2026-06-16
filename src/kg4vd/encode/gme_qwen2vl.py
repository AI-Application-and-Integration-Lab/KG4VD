"""GME-Qwen2-VL encoder.

Lazy-loads transformers / torch only when this class is instantiated, so the
core kg4vd package can be installed CPU-only.

Routes each card to the model's `get_text_embeddings` /
`get_image_embeddings` / `get_fused_embeddings` by modality and L2-normalizes
the result. Requires the GME weights and a GPU (install with
`pip install "kg4vd[gme]"`).
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from kg4vd.config.schema import EncoderCfg
from kg4vd.core.errors import EncoderError
from kg4vd.core.registry import register
from kg4vd.core.types import EvidenceCard, Query
from kg4vd.utils.images import resize_for_vlm

logger = logging.getLogger(__name__)


@register("encoder", "gme_qwen2vl")
class GMEQwen2VLEncoder:
    """Single-vector multimodal encoder using GME-Qwen2-VL.

    Pip extras: ``pip install "kg4vd[gme]"``
    """

    name = "gme_qwen2vl"

    def __init__(self, cfg: EncoderCfg):
        self.cfg = cfg
        self.dim = cfg.dim
        self.normalize = cfg.normalize
        self._model = None
        self._device = None
        self._max_long_side = int(cfg.extra.get("max_image_long_side", 1024))
        # Query instruction. Default = MegaRAG's text->image retrieval
        # instruction, so query encoding aligns with MegaRAG (whose
        # hf_gme_embed passes this for is_query=True, with pages embedded
        # image-only - see evidence_cards.page_card.image_only).
        #
        # GME-Qwen2-VL forces the default ("You are a helpful assistant.")
        # whenever is_query=False or instruction is None, so documents always
        # use that default. Tradeoff (measured, same text, doc vs query cos):
        #   query "Find an image..." -> 0.9408  text↔text (entity/relation)
        #   query None (= doc default) -> 1.0000 text↔text
        # i.e. this instruction helps text→image (page) retrieval but is
        # slightly worse for text→text. Set encoder.extra.instruction_query
        # to None for the text-text-optimal behaviour instead.
        self._instruction_query = cfg.extra.get(
            "instruction_query", "Find an image that matches the given text."
        )
        self._lazy_load()

    def _lazy_load(self) -> None:
        # Disable HF tokenizers fork-parallelism before the model (and its
        # tokenizer) load, else later forks flood stderr with the
        # "process just got forked" warning. Overridable via the env var.
        import os

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            import torch
            from transformers import AutoModel
            from transformers import logging as hf_logging
        except ImportError as e:
            raise EncoderError(
                "GMEQwen2VLEncoder requires torch + transformers. "
                "Install with: pip install \"kg4vd[gme]\""
            ) from e

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        model_id = self.cfg.model or "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct"
        logger.info("Loading GME encoder %s on %s", model_id, device)
        # Silence the spurious "weights not used / newly initialized"
        # warning that transformers >=4.52 prints when loading the GME
        # checkpoint. The Qwen2-VL refactor in transformers 4.52 moved
        # the layers from `model.layers.*` to `model.language_model.*`
        # / `model.visual.*`, so the from_pretrained key check sees a
        # mismatch -- but the GME custom modeling code (loaded via
        # trust_remote_code=True) re-maps the keys correctly afterwards.
        # Empirically: the visual.blocks LayerNorm scales come out at
        # ~5/~17 (trained values), not the all-ones init they would
        # have if the warning were truthful. Bottom line: weights ARE
        # loaded, the warning is a false alarm, and we suppress it
        # locally to avoid 50+ lines of scary log noise on every build.
        prev_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=device if device == "cuda" else None,
                trust_remote_code=True,
            ).to(device).eval()
        except Exception as e:  # noqa: BLE001
            raise EncoderError(
                f"Failed to load GME model {model_id!r}: {e!r}"
            ) from e
        finally:
            hf_logging.set_verbosity(prev_verbosity)
        self._model = model
        self._device = device

    def close(self) -> None:
        """Release the GME weights + free the GPU cache.

        Used by the staged query flow so sglang can claim the VRAM after the
        query embeddings are computed (GME-7B and the 35B LLM can't co-reside).
        """
        self._model = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    async def encode_query(self, query: Query) -> np.ndarray:
        return await self._encode(query.text, query.images, is_query=True)

    async def encode_card(self, card: EvidenceCard) -> np.ndarray:
        text = card.text_payload or ""
        imgs = [card.image_payload] if card.image_payload else []
        return await self._encode(text, imgs, is_query=False)

    async def encode_cards_batch(self, cards: list[EvidenceCard]) -> np.ndarray:
        """Embed many cards in batched GME forwards (vs one call per card).

        Cards are grouped by modality - text-only / image-only / fused - and
        each group is sent to GME in one call (GME chunks internally at
        ``embed_batch_size``, default 32). Documents (is_query=False) always
        get GME's default instruction. Results are reassembled into the
        original card order, then L2-normalized.
        """
        import torch  # noqa: WPS433

        n = len(cards)
        if n == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        texts = [c.text_payload or "" for c in cards]

        async def _resize(c: EvidenceCard) -> str | None:
            if not c.image_payload:
                return None
            return await resize_for_vlm(c.image_payload, max_long_side=self._max_long_side)

        imgs = await asyncio.gather(*[_resize(c) for c in cards])

        text_only = [i for i in range(n) if imgs[i] is None]
        image_only = [i for i in range(n) if imgs[i] is not None and not texts[i]]
        fused = [i for i in range(n) if imgs[i] is not None and texts[i]]

        bs = int(self.cfg.extra.get("embed_batch_size", 32))
        # Images go through the Qwen2-VL vision tower, whose memory scales with
        # (patches per image) x (batch). A full text batch (32) of high-res
        # page crops OOMs a 32GB card, so image/fused batches use a much
        # smaller size.
        img_bs = int(self.cfg.extra.get("embed_image_batch_size", 4))
        out = np.zeros((n, self.dim), dtype=np.float32)
        with torch.no_grad():
            if text_only:
                emb = self._model.get_text_embeddings(
                    texts=[texts[i] for i in text_only],
                    instruction=None, is_query=False, batch_size=bs,
                )
                out[text_only] = self._to_numpy(emb)
            if image_only:
                emb = self._model.get_image_embeddings(
                    images=[imgs[i] for i in image_only],
                    instruction=None, is_query=False, batch_size=img_bs,
                )
                out[image_only] = self._to_numpy(emb)
            if fused:
                emb = self._model.get_fused_embeddings(
                    texts=[texts[i] for i in fused],
                    images=[imgs[i] for i in fused],
                    instruction=None, is_query=False, batch_size=img_bs,
                )
                out[fused] = self._to_numpy(emb)

        if self.normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            out = out / norms
        return out.astype(np.float32, copy=False)

    @staticmethod
    def _to_numpy(emb) -> np.ndarray:
        import torch  # noqa: WPS433
        if hasattr(emb, "detach"):
            arr = emb.detach()
            if arr.dtype == torch.bfloat16:
                arr = arr.to(torch.float32)
            return arr.cpu().numpy()
        return np.asarray(emb)

    async def _encode(
        self, text: str, images: list[str], is_query: bool
    ) -> np.ndarray:
        import torch  # noqa: WPS433

        # Resize images for the VLM.
        resized: list[str] = []
        for p in images or []:
            if p:
                resized.append(await resize_for_vlm(p, max_long_side=self._max_long_side))

        # Docs always get the model's default instruction (GME forces it when
        # is_query=False); queries use the configured instruction (None ->
        # same default, so query and doc share one subspace).
        instruction = self._instruction_query if is_query else None

        with torch.no_grad():
            if not resized:
                emb = self._model.get_text_embeddings(
                    texts=[text or ""], instruction=instruction, is_query=is_query
                )
            elif not text:
                emb = self._model.get_image_embeddings(
                    images=resized, instruction=instruction, is_query=is_query
                )
            else:
                emb = self._model.get_fused_embeddings(
                    texts=[text], images=resized,
                    instruction=instruction, is_query=is_query,
                )
        arr = self._to_numpy(emb)
        # Some GME paths return shape (1, D); flatten.
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        v = arr.astype(np.float32, copy=False)
        if self.normalize:
            n = float(np.linalg.norm(v)) or 1.0
            v = v / n
        return v
