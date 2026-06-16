You are answering a document question from compact KG/text evidence.

Use only the evidence sketch. If the evidence is incomplete, answer only what
is supported and make uncertainty clear.

Follow the answer guidelines when they are provided. They define the requested
style, length, citation format, and level of detail.

Adapt the answer length to the question:
- For direct lookup questions, give a concise answer.
- For bridge, comparison, timeline, "walk me through", or document-level
  synthesis questions, write an evidence-rich answer that covers the relevant
  retrieved facts instead of compressing them into a short summary.
- Preserve concrete names, numbers, dates, and page-specific details when they
  appear in evidence.
- If the evidence is missing key requested parts, state the missing parts once
  and stop. Do not pad the answer or repeat unsupported absence claims.
- Cite the page numbers that support the answer.

Answer guidelines:
{{ answer_guidelines }}

Evidence sketch:
{{ text_context }}

Return strict JSON only:
{
  "answer": "final answer",
  "reasoning": "evidence-based explanation",
  "cited_pages": [page numbers],
  "confidence": "high|medium|low",
  "failure_reason": null or "not_found"
}

Question: {{ query }}
