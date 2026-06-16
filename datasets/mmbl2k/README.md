# MMLongBench-Doc 2k

This directory contains metadata and queries for the MMLongBench-Doc 2k corpus
used in KG4VD experiments.

## Files

```text
datasets/mmbl2k/
  queries.jsonl
  pdfs/
    mmlongbench_doc_2k.pdf
```

`queries.jsonl` contains 414 questions. `pdfs/` contains the 2000-page corpus
PDF with 48 MMLongBench-Doc documents.

The PDF bundle is available here:
[mmbl2k_pdfs.zip](https://drive.google.com/file/d/1mbf-vNVrCkQamtpW8tM1MUAPtLbdacSt/view?usp=sharing)

Expected SHA256:

```text
2d1fe0b32357fe0c8993303ad0e3be01107e9876ce731a6bd336a15a7553f5b3  mmbl2k_pdfs.zip
```

## Schema

Each row in `queries.jsonl` contains:

| Field | Description |
|---|---|
| `id` | question id |
| `question` | question text |
| `answer` | reference answer |
| `answer_format` | expected answer shape |
| `doc_type` | document type annotation |
| `evidence_pages` | gold evidence page ids |
