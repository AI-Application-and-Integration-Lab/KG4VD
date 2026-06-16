You are answering a document visual question.

Use only the provided page images. If the answer is visible, answer directly
and cite the page numbers you used. If the provided pages do not contain
enough evidence, say that the answer is not supported by the provided pages.

The provided images correspond to these original document page numbers, in order:
{{ pages }}

Use only these original page numbers in `cited_pages`.

Return strict JSON only:
{
  "answer": "final answer; concise for lookup questions, detailed only if the question asks for synthesis",
  "reasoning": "evidence-based explanation",
  "cited_pages": [page numbers],
  "confidence": "high|medium|low",
  "failure_reason": null or "not_found"
}

Question: {{ query }}
