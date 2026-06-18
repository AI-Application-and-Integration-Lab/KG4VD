# Test Recipe

This folder contains a small example recipe for building and querying a KG4VD
index. Start from `recipe.yaml`, then change only the fields you need.

## Recipe Metadata and Presets

Recipes can define metadata and optionally inherit preset files:

```yaml
recipe:
  name: my-document
  description: short note for this run
  extends:
    - presets/base.yaml
    - presets/encoder.gme_qwen2vl.yaml
    - presets/llm.openrouter.gpt4omini.yaml
    - presets/mineru.yaml
```

`extends` is applied in order. Later files override earlier files, and values
inside `recipe.yaml` override all presets.

Common presets:

| Preset | Purpose |
|---|---|
| `presets/base.yaml` | General defaults. |
| `presets/encoder.gme_qwen2vl.yaml` | GME-Qwen2-VL encoder and nano index. |
| `presets/encoder.mock.yaml` | Mock encoder for smoke tests. |
| `presets/llm.openrouter.gpt4omini.yaml` | OpenRouter GPT-4o-mini backend. |
| `presets/llm.openai.gpt4omini.yaml` | OpenAI GPT-4o-mini backend. |
| `presets/llm.sglang.qwen36.yaml` | Local sglang Qwen backend. |
| `presets/mineru.yaml` | MinerU parsing and visual crop settings. |

## Basic Fields

```yaml
dataset:
  name: test
  data_dir: ./recipes/test/data
  work_dir: ./recipes/test/exp
```

- `name`: any short name for the run.
- `data_dir`: folder containing the PDF files.
- `work_dir`: output folder for parsed pages, KG files, evidence cards, index,
  traces, and reports.

## Ingest

```yaml
ingest:
  parser: mineru
  dpi: 300
  jpeg_quality: 90
  max_image_long_side: 2048
```

Available parsers:

| Value | Use case |
|---|---|
| `mineru` | Recommended for visually rich PDFs. Requires the MinerU env. |
| `pypdfium` | Lightweight PDF rendering. Does not provide MinerU layout components. |

## LLM Backend

The default LLM is configured under `generator.llm`. The current test recipe
uses local sglang; replace this block if you prefer OpenRouter or OpenAI.

```yaml
generator:
  llm:
    kind: openrouter
    model: openai/gpt-4o-mini
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
```

Common options:

| `kind` | API key | Notes |
|---|---|---|
| `openrouter` | `OPENROUTER_API_KEY` | Default remote backend for most recipes. |
| `openai` | `OPENAI_API_KEY` | Direct OpenAI API. |
| `sglang` | none | Local OpenAI-compatible sglang server. |
| `vllm` | `VLLM_API_KEY` or dummy key | Local or hosted OpenAI-compatible endpoint. |
| `mock` | none | Deterministic testing only. |

For local sglang, set:

```yaml
generator:
  llm:
    kind: sglang
    model: Qwen/Qwen3.6-35B-A3B-FP8
    base_url: http://127.0.0.1:8004/v1
    api_key_env: ""
```

Stage-specific LLM fields can use `default` to reuse `generator.llm`:

```yaml
augment:
  page_summary:
    llm: default

kg:
  extract:
    llm: default

cross_page_alignment:
  judge_llm: default
```

To use a different LLM for one stage, write `<kind>:<model>`:

```yaml
augment:
  page_summary:
    llm: openrouter:openai/gpt-4o-mini

kg:
  extract:
    llm: openai:gpt-4o-mini

cross_page_alignment:
  judge_llm: sglang:Qwen/Qwen3.6-35B-A3B-FP8
```

If only a model name is provided, KG4VD treats it as an OpenRouter model:

```yaml
kg:
  extract:
    llm: openai/gpt-4o-mini
```

Supported stage-specific `kind` values are `openrouter`, `openai`, `sglang`,
`vllm`, and `mock`.

## KG Extraction

```yaml
kg:
  entity_types:
    - person
    - organization
    - date
    - scientific_concept
  visual_entity_types:
    - visual_object
    - chart_element
    - table_region
  extract:
    skip_empty_pages: true
    max_rounds: 5
    max_empty_revisions: 2
  description_merge: concat_sep
  description_max_chars: 4000
```

- `entity_types` and `visual_entity_types` define the vocabulary used by the KG
  extractor. Keep the list small and relevant to your document type.
- `max_rounds` controls how many refinement rounds are allowed per page.
- `description_merge` can be `concat_sep` or `llm_summarize`.

## Cross-page Alignment

```yaml
cross_page_alignment:
  enabled: true
  candidate_top_k: 10
  rs_threshold: 9.0
  judge_llm: default
  canonicalize_same_as: true
```

- Set `enabled: false` for a faster page-local build.
- Larger `candidate_top_k` checks more entity pairs and costs more LLM calls.
- `rs_threshold`, `rs_threshold_same_as`, `rs_threshold_rescue`, and
  `rs_threshold_cross_modality` are relevance score thresholds on a 0 to 10
  scale.

## Evidence Cards and Encoder

```yaml
evidence_cards:
  emit:
    page: true
    entity: true
    relation: true
  entity_card:
    use_visual_crop: true

encoder:
  name: gme_qwen2vl
  model: Alibaba-NLP/gme-Qwen2-VL-7B-Instruct
  dim: 3584
  normalize: true

index:
  backend: nano
  metric: cosine
```

Available index options:

| Field | Values |
|---|---|
| `index.backend` | `nano`, `faiss`, `qdrant` |
| `index.metric` | `cosine`, `dot`, `maxsim` |

The tested release recipe uses `gme_qwen2vl` with `nano` and `cosine`.

## Query Settings

```yaml
query:
  propagation: qgge
  retrieval:
    anchor_size: 40
    token_budget: 8000
  qgge:
    rounds: 2
    page_beam: 50
    lambda_page: 0.7
  reranker:
    enabled: true
```

Available query options:

| Field | Values | Notes |
|---|---|---|
| `query.propagation` | `qgge`, `ppr` | `qgge` = Query-Guided Graph Expansion. `ppr` = Personalized PageRank. |
| `query.answer.mode` | `auto`, `images`, `texts`, `fusion` | Answer context format. |
| `query.reranker.enabled` | `true`, `false` | Enables the optional Qwen3-VL reranker service. |

Use `ppr` only after building the optional `retrieval_graph` stage.

## Runtime and Logging

```yaml
obs:
  tracer: both
  rich_progress: true

runtime:
  max_async:
    extract: 12
    align: 12
    embed: 16
```

Available tracer values: `jsonl`, `rich`, `both`, `none`.

Increase `runtime.max_async` for faster builds if your LLM backend can handle
more concurrent requests. Lower it if you see rate limits, timeouts, or GPU
memory pressure.

## Validate Changes

After editing `recipe.yaml`, inspect the resolved config:

```bash
conda activate kg4vd
kg4vd show-config recipes/test/recipe.yaml
```

Build the recipe:

```bash
kg4vd build recipes/test/recipe.yaml --resume
```

You can also override a value without editing the file:

```bash
kg4vd build recipes/test/recipe.yaml --set query.reranker.enabled=false
```
