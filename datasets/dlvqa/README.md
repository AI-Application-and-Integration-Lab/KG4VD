# DLVQA

DLVQA is a document-level visual question answering benchmark for long,
visually rich documents. It focuses on questions that require global document
comprehension rather than single-page lookup.

The release contains 525 open-ended questions over four documents. Each example
includes reference evidence, topic-level guidance, and answer-format metadata.

## Files

```text
datasets/dlvqa/
  queries.jsonl
  pdfs/
    DLCV.pdf
    World_History_Volume_1.pdf
    environmental-report.pdf
    picture_books.pdf
```

`queries.jsonl` contains all benchmark questions. `pdfs/` contains the four
source documents.

The PDF bundle is available here:
[dlvqa_pdfs.zip](https://drive.google.com/file/d/1erOxT735LCF_VUnlR5eKJ_Mki6cTliZY/view?usp=sharing)

Expected SHA256:

```text
ade067b96d52cfa5d8f151bc1e09ce72ac29058e7aa207bf8d6a010306d0346a  dlvqa_pdfs.zip
```

## Schema

Each JSONL row contains:

| Field | Description |
|---|---|
| `qa_id` | question id |
| `benchmark` | benchmark name, `dlvqa` |
| `subset`, `subset_slug` | document subset identifiers |
| `doc_id` | document id |
| `query` | question text |
| `gt_answer` | optional reference answer |
| `gt_facts` | supporting facts with `id`, `text`, `page`, and `type` |
| `gt_topics` | topic outline for document-level coverage |
| `answer_guidelines` | generation and citation instructions |
| `page_span` | pages relevant to the question |
| `answer_format` | `prose` or `wiki` |
| `extra` | additional metadata |
