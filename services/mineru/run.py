"""MinerU 3.x parse runner - invoked out-of-process by the kg4vd ingest stage.

MinerU 3.x requires transformers>=4.57.3, which conflicts with the GME encoder's
transformers<4.52 pin, so MinerU lives in its own conda env and kg4vd shells out
to this runner. The runner uses MinerU's Python API
(`mineru.cli.common.do_parse`) directly and emits only what the pipeline needs
(middle.json + image crops).

Output layout (backend=pipeline, parse_method=auto):
    <out_dir>/<stem>/auto/<stem>_middle.json
    <out_dir>/<stem>/auto/images/

Usage (run with the MinerU env's python):
    python services/mineru/run.py --pdf <file.pdf> --out <out_dir> [--lang en] [--backend pipeline]
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--backend", default="pipeline")
    ap.add_argument("--method", default="auto")
    args = ap.parse_args()

    from mineru.cli.common import do_parse, read_fn

    pdf_path = Path(args.pdf)
    stem = pdf_path.stem
    pdf_bytes = read_fn(pdf_path)

    do_parse(
        output_dir=args.out,
        pdf_file_names=[stem],
        pdf_bytes_list=[pdf_bytes],
        p_lang_list=[args.lang],
        backend=args.backend,
        parse_method=args.method,
        formula_enable=True,
        table_enable=True,
        # The pipeline only needs middle.json + image crops; skip the rest.
        f_dump_middle_json=True,
        f_dump_md=False,
        f_dump_content_list=False,
        f_dump_model_output=False,
        f_dump_orig_pdf=False,
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
    )

    middle = Path(args.out) / stem / args.method / f"{stem}_middle.json"
    if not middle.is_file():
        raise SystemExit(f"MinerU finished but {middle} was not produced.")
    print(f"OK {middle}")


if __name__ == "__main__":
    main()
