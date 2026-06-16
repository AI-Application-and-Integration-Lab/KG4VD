"""Prompt templates for KG construction.

All prompts return a versioned constant so the cache key naturally invalidates
when we edit a prompt.

The output schema is JSON-only and matches the parser in
``kg4vd.kg.extract.parse_extraction_output``.
"""

from __future__ import annotations

from textwrap import dedent

PROMPTS_VERSION = "kg4vd-v1.0.0"

ENTITY_EXTRACTION_INIT = dedent(
    """
    You are extracting a multimodal knowledge graph from a single page of a
    visually rich document. The page contains text and may contain figures,
    charts, diagrams, or tables.

    Goal: identify entities (text and visual) and relations between them.

    ENTITY TYPES (text):
    {entity_types}

    VISUAL ENTITY TYPES:
    {visual_entity_types}

    VISUAL DESCRIPTION RULES - IMPORTANT.
    For every entity with modality="visual", the `visual_description`
    field must capture details that are ONLY available from the page
    image and that text-only extraction would miss. Generic phrases such
    as "rectangular area at the bottom of the page", "section in the
    middle", or "image showing the topic" are INSUFFICIENT and will be
    rejected on review.

    Aim for 1-3 specific sentences. Where applicable, include:
      - shape and size - e.g. "tall narrow column spanning ~30% of page
        width", "full-width banner ~3cm tall", "circle with three
        concentric rings".
      - colour palette - e.g. "navy header on white background", "red
        bar next to four grey bars", "yellow highlight on the key dates".
      - typography - serif vs sans-serif, bold/italic, relative size
        (title vs caption), distinctive font choice if any.
      - position relative to nearby anchors - e.g. "directly above the
        table titled X", "to the left of the histogram aligned with the
        y-axis", NOT just "bottom of the page".
      - layout role - section divider, footer, callout box, sidebar,
        logo banner, icon row, etc.

    Per visual_type, also add:
      - chart_element        : chart type (bar/line/pie/scatter/...),
                               axis labels visible, number of
                               series/bars, relative magnitudes
                               ("rightmost bar ~3x taller than leftmost"),
                               legend contents.
      - diagram_component    : the components, arrows, connection
                               topology, which nodes are labelled, the
                               diagram's overall structure
                               (tree / cycle / pipeline / matrix / ...).
      - table_region         : row x column counts, header style, merged
                               cells if any, dominant content type
                               (numbers / dates / names / icons).
      - figure_panel         : depicted subject, composition, colour
                               mood, presence of people / objects /
                               scenery, photo vs illustration.
      - layout_region        : the bounding role (banner / sidebar /
                               margin / body / footer) AND a digest of
                               what is visually inside - e.g. "list of
                               32 small photo thumbnails arranged in 4
                               rows of 8, each with a name caption
                               below", "two-column logo+paragraph block
                               with the OpenStax wordmark in the upper
                               left corner".
      - visual_object        : the object itself (logo, icon, glyph,
                               sticker), colour, shape, any text it
                               contains.

    Counter-examples (do NOT do this):
      visual_description: "rectangular area at the bottom of the page"
      visual_description: "image of the topic"
      visual_description: "section that lists names"

    Good examples:
      "Three-column footer ~2cm tall in dark grey on white, containing
       publisher contact info; the OpenStax wordmark sits left, a row of
       social icons centre, and the page number right-aligned."
      "Vertical bar chart with 5 navy bars and one red bar (the
       rightmost), x-axis labelled 1990-2020 in 5-year ticks, y-axis
       'GDP growth (%)'. Red bar (~6%) is roughly 3x taller than the
       smallest navy bar (~2%)."
      "Block of 32 portrait thumbnails in a 4x8 grid, each ~3cm square,
       captioned with the contributor's name in small serif italics
       below; thumbnails are colour photographs on a white background."

    ENTITY DISAMBIGUATION RULES - IMPORTANT.

    1. PERSON PHOTOS. If a page shows a portrait photograph of a real
       person, create TWO separate nodes:
         - one with modality="text" and entity_type="person"
           describing the person in the abstract (their role,
           affiliation, contributions);
         - one with modality="visual",
           visual_type="visual_object" (or "figure_panel" if the
           photograph is one cell of a larger panel), describing the
           PHOTOGRAPH itself,
       and connect them with a "depicts" edge from the visual node
       to the person node. NEVER produce a single node with
       modality="visual" + entity_type="person" - that combination is
       a contradiction and will be rejected on review.

    2. LAYOUT CHROME IS NOT AN ENTITY. Page numbers ("Page 1 of 5"),
       running headers, page-edge folio markers, and similar layout
       chrome are not entities. Do not extract them.

    3. AVOID SPECULATION. Describe only what you can verify from the
       image. If you cannot resolve a face, say "small portrait
       photographs" rather than "blurred faces"; if you cannot tell a
       colour for sure, say "muted background" rather than guessing a
       specific shade. The text "blurred", "faded", "unclear" must
       describe a real visual property of the page, not a limitation
       of your own perception.

    4. GROUPS WITH NAMED MEMBERS. A heading like "Reviewers",
       "Contributing Authors", or "Editorial Board" that introduces a
       list of named people is itself a "group" entity
       (entity_type="group"), NOT an "event". Each named person under
       it is a separate "person" entity, plus an `includes` edge from
       the group to each person.

    OUTPUT - JSON only, no markdown, no commentary. Schema:
    {{
      "nodes": [
        {{
          "name": "<short canonical name>",
          "entity_type": "<one of the types listed above>",
          "modality": "text" | "visual",
          "description": "<full sentence in language of the document>",
          "visual_description": "<1-3 sentences obeying the rules above; only for modality=visual>",
          "visual_type": "<one of the visual types; only for modality=visual>"
        }}
      ],
      "edges": [
        {{
          "src": "<exact name of head node>",
          "tgt": "<exact name of tail node>",
          "relation": "<verb phrase or label>",
          "description": "<1-2 sentence rationale grounded in this page>",
          "visual_evidence_hint": "<short ref to a figure/label if relevant>",
          "confidence": <float in [0,1]>
        }}
      ]
    }}

    EXAMPLE (for reference; the actual page below is different).
    Suppose the page (text + image) shows:
        "Founded in 1886, MegaCorp was established by John Smith in
         Chicago. The company specialises in widgets. Figure 1 shows
         the company logo: a blue gear surrounded by three concentric
         rings, with the wordmark 'MegaCorp' below."
    A good extraction is:
    {{
      "nodes": [
        {{"name":"MegaCorp","entity_type":"organization","modality":"text",
          "description":"MegaCorp is a widget manufacturer founded in 1886 by John Smith in Chicago."}},
        {{"name":"John Smith","entity_type":"person","modality":"text",
          "description":"John Smith is the founder of MegaCorp."}},
        {{"name":"Chicago","entity_type":"location","modality":"text",
          "description":"Chicago is the city where MegaCorp was founded."}},
        {{"name":"1886","entity_type":"date","modality":"text",
          "description":"The year MegaCorp was founded."}},
        {{"name":"MegaCorp logo","entity_type":"visual_object","modality":"visual",
          "visual_type":"visual_object",
          "description":"The MegaCorp logo shown in Figure 1.",
          "visual_description":"Blue gear ~3cm wide surrounded by three concentric rings of decreasing thickness; sans-serif 'MegaCorp' wordmark in dark blue centred below the gear."}}
      ],
      "edges": [
        {{"src":"MegaCorp","tgt":"John Smith","relation":"founded by",
          "description":"The page states MegaCorp was established by John Smith.","confidence":0.95}},
        {{"src":"MegaCorp","tgt":"1886","relation":"founded in",
          "description":"The page gives 1886 as MegaCorp's founding year.","confidence":0.95}},
        {{"src":"MegaCorp","tgt":"Chicago","relation":"located in",
          "description":"The page states MegaCorp was established in Chicago.","confidence":0.9}},
        {{"src":"John Smith","tgt":"Chicago","relation":"founded MegaCorp in",
          "description":"John Smith established MegaCorp in Chicago.","confidence":0.85}},
        {{"src":"MegaCorp logo","tgt":"MegaCorp","relation":"depicts",
          "description":"Figure 1 is the visual emblem of MegaCorp.",
          "visual_evidence_hint":"Figure 1: blue gear + concentric rings","confidence":0.95}}
      ]
    }}
    Notice five edges from four small facts: the model emits a relation
    for every pair the text actually supports, includes a cross-modality
    `depicts` edge from the logo to the organisation, and the visual
    entity has a concrete visual_description. Aim for similarly dense
    edges on the real page below.

    Page number: {page_id} of {total_pages}
    {page_text_block}
    """
).strip()
