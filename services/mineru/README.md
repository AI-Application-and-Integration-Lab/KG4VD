# services/mineru

Out-of-process **MinerU 3.x** PDF parsing for the ingest stage.

MinerU 3.x requires `transformers>=4.57.3`, which conflicts with the GME
encoder's `transformers<4.52` pin - so MinerU cannot live in the `kg4vd` env.
It runs in its **own conda env**, and the ingest stage shells out to `run.py`,
which calls MinerU's Python API (`mineru.cli.common.do_parse`) directly - not
the `magic-pdf` / `mineru` CLI binary.

## Setup (one-time)

```bash
conda create -n mineru python=3.11 -y
conda activate mineru
uv pip install -U "mineru[pipeline]"     # pipeline backend (layout → middle.json)
# models download to $HF_HOME on first parse
```

`[pipeline]` is the layout-analysis backend matching the construction pipeline
(blocks → `middle.json`). `[all]` additionally pulls the VLM backend / vllm.

## How ingest uses it

`src/kg4vd/ingest/mineru_parser.py::_run_mineru` invokes:

```bash
<MINERU_PYTHON> services/mineru/run.py --pdf <file.pdf> --out <out_dir> --lang en
```

- `MINERU_PYTHON` - path to the MinerU env's python
  (default `<conda>/envs/mineru/bin/python` via scripts/paths.sh; override per box).
- `KG4VD_MINERU_RUNNER` - path to this `run.py` (default `services/mineru/run.py`).

Output (backend=pipeline, parse_method=auto), consumed unchanged by the parser:

```
<out_dir>/<stem>/auto/<stem>_middle.json
<out_dir>/<stem>/auto/images/
```

## Run directly

```bash
conda activate mineru
python services/mineru/run.py \
    --pdf doc.pdf --out /tmp/out --lang en
```
