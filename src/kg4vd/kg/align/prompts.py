"""Cross-page alignment judge prompts.

Co-located with the aligner (kg/align/cross_page.py). The shared prompt-set
version stamp lives in kg/prompts.py (PROMPTS_VERSION).
"""

from __future__ import annotations

from textwrap import dedent


CROSS_PAGE_ALIGN_JUDGE = dedent(
    """
    You are judging cross-page entity pairs. Per-page extraction produced
    entities on each page in isolation; many real facts span pages and
    were therefore missed. Your job is to recover them:

      - decide if SOURCE and CANDIDATE are the SAME real entity, OR
      - name the semantic relation between them (free-form), OR
      - say they are unrelated.

    SOURCE:
      name:         {src_name}
      entity_type:  {src_entity_type}
      modality:     {src_modality}{src_visual_block}
      description:  {src_description}
      pages:        {src_pages}
      page_context:
    {src_page_context}

    CANDIDATES:
    {candidates_block}

    For EACH candidate emit exactly one decision:

      - "same_as":   SOURCE and CANDIDATE refer to the SAME real-world
                     entity (e.g. "OpenStax" vs "openstax.org"). Hard
                     bar: same entity_type, same role/scope, no
                     contradicting attributes. Triggers a graph-level
                     merge - keep this strict.

      - "related":   not the same entity, but a meaningful semantic
                     relation holds. You MUST supply a `relation`
                     string in `snake_case` verb-phrase form, using
                     the SAME vocabulary style the per-page extractor
                     uses. Examples:
                       "is_sibling_of", "depicts", "illustrates",
                       "part_of", "authored_by", "located_in",
                       "appears_in_figure", "released_under",
                       "supports_claim", "follows", "caused_by".
                     The relation reads SOURCE -> CANDIDATE; choose
                     the verb so the direction reads naturally
                     (e.g. for a chart element grounding a concept,
                     SOURCE is the visual and relation is "depicts").

      - "no":        not the same entity and no clear relation.

    Hard rules:
      1. "same_as" requires matching entity_type. If types differ,
         pick "related" with an appropriate verb (organisation vs
         license → "released_under"; figure vs concept → "depicts";
         section vs publisher → "no"), never "same_as".
      2. Do NOT use generic containment ("is_part_of", "part_of",
         "supports_claim", "is_related_to", "includes") as a default
         when the candidate is the document, the report, a top-level
         initiative, or any whole-document concept. Co-occurrence
         in the same publication is NOT a relation. Concretely
         BAD (pick "no"):
           - "carbon neutrality"  is_part_of "2024 Environmental Progress Report"  → no
           - "Apple Watch"        is_part_of "2024 Environmental Progress Report"  → no
           - "Chapter 14"         is_part_of "OpenStax"                            → no
           - any X                supports_claim  "the report"                    → no
         Only emit "is_part_of" / "part_of" when the part-of fact is
         a genuine specific structural claim (e.g. an item literally
         in a labelled list, a sub-component of a labelled assembly).
      3. Reuse a relation name that the per-page extractor would
         have used. Prefer concise verb phrases ("part_of") over
         wordy ones ("is_a_part_of_the").
      4. Never name a relation more abstract than the evidence
         supports. If you can't tell from the page_context what the
         relation actually is, pick "no", not "is_related_to".

    Confidence rs in 0..10:
      - "same_as":  rs >= {rs_threshold_same_as}.
      - "related":  rs >= {rs_threshold_floor}.
      Below floor → omit the candidate entirely. (Downstream may
      apply a stricter post-filter on weak "related" edges; emit
      whatever meets the floor and let the system decide.)

    OUTPUT - JSON only:
    [
      {{ "candidate": "<exact candidate name>",
         "decision":  "same_as" | "related" | "no",
         "relation":  "<snake_case verb, required when decision='related'>",
         "rs":        <int 0..10>,
         "rationale": "<1 sentence>" }}
    ]
    """
).strip()


CROSS_PAGE_ALIGN_JUDGE_VISUAL = dedent(
    """
    You are judging cross-page entity pairs. Per-page extraction produced
    entities on each page in isolation; many real facts span pages and
    were therefore missed. Your job is to recover them:

      - decide if SOURCE and CANDIDATE are the SAME real entity, OR
      - name the semantic relation between them (free-form), OR
      - say they are unrelated.

    You have BOTH text descriptions AND attached page-region images
    cropped from the original PDF. Use the images as a second channel:
    visual identity, layout context, and surrounding text in the crop
    can confirm or contradict the text-only descriptions.

    ATTACHED IMAGES (in order, before this text):
    {image_index_block}

    SOURCE:
      name:         {src_name}
      entity_type:  {src_entity_type}
      modality:     {src_modality}{src_visual_block}
      description:  {src_description}
      pages:        {src_pages}
      page_context:
    {src_page_context}

    CANDIDATES:
    {candidates_block}

    For EACH candidate emit exactly one decision:

      - "same_as":   SOURCE and CANDIDATE refer to the SAME real-world
                     entity. Hard bar: same entity_type, same role/scope,
                     no contradicting attributes. For visual entities,
                     the crops should depict the SAME visual subject
                     (allowing for redrawing, re-sized, slightly
                     different pose). Different artworks of the same
                     character → same_as.

      - "related":   not the same entity, but a meaningful semantic
                     relation holds. You MUST supply a `relation`
                     string in `snake_case` verb-phrase form.

      - "no":        not the same entity and no clear relation.

    Hard rules:
      1. "same_as" requires matching entity_type.
      2. Do NOT use generic containment as a default.
      3. Reuse a relation name a per-page extractor would use.
      4. Never name a relation more abstract than the evidence supports.
      5. **If the attached image clearly contradicts the text-only
         description** (e.g. text says "butterfly" but the cropped
         region shows a panel of monsters), LOWER the rs score and
         say so in `rationale`. The crop's bbox is at the layout-
         component level, so an entity may be one of several subjects
         in the panel - use the surrounding visual context to judge.
      6. **If the attached images clearly show the SAME visual subject**
         even when text descriptions differ in wording, lean toward
         same_as / related (whichever fits the entity_type rule).

    Confidence rs in 0..10:
      - "same_as":  rs >= {rs_threshold_same_as}.
      - "related":  rs >= {rs_threshold_floor}.
      Below floor → omit the candidate entirely.

    OUTPUT - JSON only:
    [
      {{ "candidate": "<exact candidate name>",
         "decision":  "same_as" | "related" | "no",
         "relation":  "<snake_case verb, required when decision='related'>",
         "rs":        <int 0..10>,
         "rationale": "<1 sentence, MUST mention what the image showed
                      when an image was attached for this candidate>" }}
    ]
    """
).strip()
