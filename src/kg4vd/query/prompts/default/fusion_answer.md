You are given two draft answers to the same document question.

The image answer is grounded in page images. The text answer is grounded in
KG/text evidence. Prefer directly visible page evidence for single visual
details, and use KG/text evidence for bridges or document-level context.
If the drafts conflict, choose the better supported answer and explain briefly.
If the drafts are mostly compatible but use different labels for the same
visual item, preserve both labels in a compact form instead of discarding one
(for example, "square/rectangle" or "square-like rectangle"). Do not invent a
new unsupported label.
For direct lookup questions, keep the final answer concise. For synthesis
questions, preserve the useful coverage from both drafts.

Question: {{ query }}

Image answer:
{{ image_draft }}

Text answer:
{{ text_draft }}

Return strict JSON only:
{
  "answer": "final answer",
  "reasoning": "evidence-based explanation",
  "cited_pages": [page numbers],
  "confidence": "high|medium|low",
  "failure_reason": null or "not_found"
}
