Rank the candidate document pages by usefulness for answering the question.

Return strict JSON only:

```json
{"page_ids": [3, 8, 2]}
```

Rules:

- Use only candidate page ids.
- Put the most useful pages first.
- Prefer pages likely to contain exact visual or textual evidence.
- Do not include markdown.

Question:
{{ query }}

Candidate pages:
{{ pages }}
