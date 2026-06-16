"""Query entry points - staged so GME and the LLM never share the GPU.

GME-7B (query embedding) and the 35B LLM server can't co-reside on a 32GB card,
but a query only needs GME for the anchor encode. So the flow is:

  1. encode ALL query vectors with GME on the GPU, then free GME;
  2. start the sglang server (now the GPU is free) and answer every query with
     the LLM, feeding the precomputed vectors (no GME needed).

``build_query_pipeline`` is exposed for tests/batch drivers (inject mock
LLM/encoder). ``run_query`` / ``run_query_batch`` do the staged orchestration.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
import urllib.request
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from kg4vd.config.schema import KG4VDConfig
from kg4vd.core.types import Query, QueryResult
from kg4vd.encode import build_encoder
from kg4vd.llm.factory import build_llm
from kg4vd.obs import Tracer
from kg4vd.obs.tracer import stage
from kg4vd.query.anchors import AnchorRetriever
from kg4vd.query.analyzer import QueryAnalyzer
from kg4vd.query.context import ContextBuilder, TextItemBuilder
from kg4vd.query.generator import AnswerGenerator
from kg4vd.query.loader import QueryArtifacts, load_query_artifacts
from kg4vd.query.pipeline import QueryPipeline
from kg4vd.query.ppr import PPRPropagator
from kg4vd.query.prompts import PromptSet
from kg4vd.query.propagation import QGGEPropagator
from kg4vd.query.reranker import build_reranker

logger = logging.getLogger(__name__)


def build_propagator(cfg: KG4VDConfig, artifacts: QueryArtifacts):
    """Pick the graph-propagation method from ``cfg.query.propagation``.

    Both return the uniform ``propagate(anchors, query_vector) -> (hits,
    node_scores)`` contract. PPR needs the retrieval graph.
    """
    if cfg.query.propagation == "ppr":
        if artifacts.retrieval_graph is None:
            raise RuntimeError(
                "query.propagation=ppr requires kg/retrieval_graph.json; build it "
                "with `kg4vd build --stages retrieval_graph --resume`."
            )
        return PPRPropagator(artifacts.retrieval_graph, artifacts.index, cfg.query.ppr)
    return QGGEPropagator(artifacts.index, cfg.query.qgge)


def build_query_pipeline(
    cfg: KG4VDConfig, artifacts: QueryArtifacts, *, llm: Any, encoder: Any
) -> QueryPipeline:
    prompts = PromptSet(cfg.query.answer.prompt_set)
    return QueryPipeline(
        analyzer=QueryAnalyzer(llm, prompts),
        anchors=AnchorRetriever(artifacts.index, encoder),
        propagator=build_propagator(cfg, artifacts),
        text_builder=TextItemBuilder(artifacts.index, artifacts.pages_by_id),
        reranker=build_reranker(cfg),
        context=ContextBuilder(token_budget=cfg.query.retrieval.token_budget),
        generator=AnswerGenerator(llm, prompts),
        cfg=cfg,
    )


# ---------------------------------------------------------------------------
# Phase 1 - GME query embeddings (on GPU), then free GME
# ---------------------------------------------------------------------------


async def encode_queries(cfg: KG4VDConfig, texts: list[str]) -> list[np.ndarray]:
    """Encode query texts with GME, then unload it to free the GPU."""
    encoder = build_encoder(cfg.encoder)
    try:
        vecs = [
            np.asarray(
                await encoder.encode_query(Query(text=t)), dtype=np.float32
            ).reshape(-1)
            for t in texts
        ]
    finally:
        close = getattr(encoder, "close", None)
        if close:
            close()
    return vecs


# ---------------------------------------------------------------------------
# Phase 2 - sglang server lifecycle (start after GME freed, stop at the end)
# ---------------------------------------------------------------------------


def _sglang_ready(base_url: str, timeout: float = 3.0) -> bool:
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


@contextmanager
def _managed_sglang(cfg: KG4VDConfig, *, manage: bool) -> Iterator[None]:
    base_url = cfg.generator.llm.base_url or "http://127.0.0.1:8004/v1"
    if not manage:
        if not _sglang_ready(base_url):
            raise RuntimeError(
                f"sglang not reachable at {base_url} and --no-manage-sglang set; "
                f"start it first (scripts/launch_qwen36_sglang.sh)."
            )
        yield
        return
    if _sglang_ready(base_url):
        # Already up (and GME must already be unloaded to have reached here).
        logger.info("sglang already running at %s; reusing it.", base_url)
        yield
        return

    script = os.environ.get(
        "KG4VD_SGLANG_LAUNCH", "scripts/launch_qwen36_sglang.sh"
    )
    if not Path(script).is_file():
        raise RuntimeError(
            f"sglang launch script not found: {script}. Set KG4VD_SGLANG_LAUNCH "
            f"or run with --no-manage-sglang and start the server yourself."
        )
    logger.info("Starting sglang server (%s) - this can take several minutes…", script)
    proc = subprocess.Popen(  # noqa: S603
        ["bash", script], start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("sglang server exited during startup.")
            if _sglang_ready(base_url):
                logger.info("sglang ready at %s", base_url)
                break
            time.sleep(5)
        else:
            raise RuntimeError("sglang server did not become ready in 15 min.")
        yield
    finally:
        logger.info("Stopping sglang server.")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Phase 1b - reranker server lifecycle (up during retrieval, down before sglang)
# ---------------------------------------------------------------------------


def _reranker_ready(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as r:  # noqa: S310
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


@contextmanager
def _managed_reranker(cfg: KG4VDConfig, *, manage: bool) -> Iterator[None]:
    url = (
        cfg.query.reranker.url
        or os.environ.get("KG4VD_RERANKER_URL")
        or "http://127.0.0.1:8003"
    )
    if not manage:
        if not _reranker_ready(url):
            raise RuntimeError(
                f"reranker not reachable at {url} and --no-manage-reranker set; "
                f"start it first (scripts/launch_reranker.sh)."
            )
        yield
        return
    if _reranker_ready(url):
        logger.info("reranker already running at %s; reusing it.", url)
        yield
        return

    script = os.environ.get("KG4VD_RERANKER_LAUNCH", "scripts/launch_reranker.sh")
    if not Path(script).is_file():
        raise RuntimeError(
            f"reranker launch script not found: {script}. Set KG4VD_RERANKER_LAUNCH "
            f"or run with --no-manage-reranker and start the server yourself."
        )
    logger.info("Starting reranker server (%s) …", script)
    proc = subprocess.Popen(  # noqa: S603
        ["bash", script], start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("reranker server exited during startup.")
            if _reranker_ready(url):
                logger.info("reranker ready at %s", url)
                break
            time.sleep(5)
        else:
            raise RuntimeError("reranker server did not become ready in 10 min.")
        yield
    finally:
        logger.info("Stopping reranker server.")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


async def run_query_batch(
    cfg: KG4VDConfig,
    items: list[tuple[str, str]],
    *,
    manage_sglang: bool = True,
    manage_reranker: bool = True,
) -> list[QueryResult]:
    """Answer many ``(qid, question)`` items with the staged flow:
    GME encode → (reranker: retrieve+rerank) → sglang generate. Each stage owns
    the GPU alone - the reranker server and sglang never run concurrently."""
    if not items:
        return []
    work_dir = Path(cfg.dataset.work_dir)
    run_id = uuid.uuid4().hex[:12]
    tracer = Tracer(
        run_id=run_id, trace_id=run_id,
        jsonl_path=work_dir / cfg.obs.trace_path,
        rich_progress=False,
        write_jsonl=cfg.obs.tracer in {"jsonl", "both"},
    )
    results: list[QueryResult] = []
    with tracer:
        # Phase 1: GME embeddings (GPU), then GME is unloaded.
        with stage("query.encode", n_queries=len(items)):
            vectors = await encode_queries(cfg, [q for _, q in items])
        artifacts = await load_query_artifacts(cfg)
        queries = [
            Query(text=question, qid=qid) if qid else Query(text=question)
            for qid, question in items
        ]

        # Phase 1b: when the reranker is on, retrieve + rerank now (GPU-free in
        # this process; reranker server holds its own GPU, sglang not yet up).
        precomputed: list[Any] | None = None
        if cfg.query.reranker.enabled:
            with _managed_reranker(cfg, manage=manage_reranker):
                retr = build_query_pipeline(cfg, artifacts, llm=None, encoder=None)
                precomputed = []
                with stage("query.retrieve", n_queries=len(queries)):
                    for q, vec in zip(queries, vectors):
                        _, ranked, node_scores, _ = await retr.retrieve(
                            q, query_vector=vec
                        )
                        precomputed.append((ranked, node_scores))

        # Phase 2: generation. Only manage a local sglang server when the LLM
        # actually is one - remote LLMs (openrouter / openai) need no local
        # server, so skip the staging entirely (no boot, no GPU handoff).
        needs_sglang = cfg.generator.llm.kind == "sglang"
        sglang_ctx = (
            _managed_sglang(cfg, manage=manage_sglang)
            if needs_sglang else nullcontext()
        )
        with sglang_ctx:
            llm = build_llm(cfg.generator.llm)
            try:
                pipe = build_query_pipeline(cfg, artifacts, llm=llm, encoder=None)
                for i, (q, vec) in enumerate(zip(queries, vectors)):
                    pc = precomputed[i] if precomputed is not None else None
                    results.append(
                        await pipe.answer(q, query_vector=vec, precomputed=pc)
                    )
            finally:
                close = getattr(llm, "aclose", None)
                if close:
                    await close()
    return results


async def run_query(
    cfg: KG4VDConfig, question: str, *, qid: str | None = None,
    manage_sglang: bool = True, manage_reranker: bool = True,
) -> QueryResult:
    """Answer one question (staged GME → reranker → sglang)."""
    results = await run_query_batch(
        cfg, [(qid or "", question)],
        manage_sglang=manage_sglang, manage_reranker=manage_reranker,
    )
    return results[0]
