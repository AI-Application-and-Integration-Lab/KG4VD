"""Prompts for the component-cued adaptive extractor.

Three prompts with a clean writer/reflector split:

  - ``EXTRACTOR_INIT_PROMPT``    - round-0 writer. Builds on the shared
    ``ENTITY_EXTRACTION_INIT`` body, patching in the ``source_components``
    field on both node and edge JSON schemas, then wraps the flat output
    in the ``ops`` shape.
  - ``REFLECTOR_PROMPT``         - coverage judge. Emits a
    ``RevisionBrief``; never writes ops itself. Includes three
    illustrative examples (covered → STOP, real gap, enumeration
    density mismatch).
  - ``EXTRACTOR_REVISE_PROMPT``  - revise-round writer. Sees the
    annotated image (with saturation overlays), manifest, scorecard,
    and the latest ``RevisionBrief``. Includes two examples (general
    revise + stub-upgrade via replace_nodes).

The prompts are ``.format(**kwargs)``-rendered; required placeholders per
prompt are listed at the top of each definition. They live next to the
extractor logic that consumes them; the shared base prompt and
``PROMPTS_VERSION`` stay in ``kg4vd.kg.prompts``.
"""

from __future__ import annotations

from textwrap import dedent

from kg4vd.kg.prompts import ENTITY_EXTRACTION_INIT as _INIT_BASE


PROMPTS_VERSION_COMPONENT_CUED = "kg4vd-component-cued-v1"


# ---------------------------------------------------------------------------
# Shared preamble - describes the annotated image + grounding contract
# ---------------------------------------------------------------------------

_ANNOTATED_IMAGE_BLOCK = dedent(
    """
    The page image is ANNOTATED. Each detected layout component is
    outlined and labelled with a coloured circle carrying a stable
    component ID:

        T*   - title / heading
        P*   - paragraph / body text
        L*   - list / table-of-contents / index
        IM*  - image / chart / figure / diagram
        TB*  - table
        F*   - formula / equation

    The circles sit in the page margin (or above the box). Colours:
    BLUE for text, RED for visual content, GREEN for table / formula.
    The boxes and labels are NAVIGATION AIDS only - do not extract them
    as entities. The actual evidence is the content inside each box.

    A compact component manifest follows at the end listing each
    labelled component with type, position, neighbours, and (for
    textual components) the full OCR text.
    """
).strip()


_GROUNDING_BLOCK = dedent(
    """
    GROUNDING REQUIREMENT. Every node and every edge MUST carry a
    non-empty `source_components` field listing the component ID(s)
    that support the claim. One component can yield many entities
    (a paragraph listing 19 contributors → 19 person entities, all
    citing that one component). Grounding is metadata, not a filter.

    TEXT INSIDE VISUAL COMPONENTS. Labelled text inside a chart /
    diagram / figure / table cell is grounded to the visual component
    (IM* / TB* / F*) that contains it, NOT to a non-existent text
    component.
    """
).strip()


# ---------------------------------------------------------------------------
# 1. Extractor - INIT prompt (Round 0)
# ---------------------------------------------------------------------------

# The init prompt reuses the base prompt verbatim and patches in
# `source_components` on both the node and edge JSON schemas. The
# strings below MUST appear exactly once each in the base prompt - if the
# base prompt is edited and these markers drift, ``_patch_init_schema``
# raises in-process and the loader fails fast.

_NODE_SCHEMA_OLD = (
    '"visual_type": "<one of the visual types; only for modality=visual>"'
)
_NODE_SCHEMA_NEW = (
    '"visual_type": "<one of the visual types; only for modality=visual>",\n'
    '          "source_components": ["<component_id>", ...]'
)
_EDGE_SCHEMA_OLD = '"confidence": <float in [0,1]>'
_EDGE_SCHEMA_NEW = (
    '"confidence": <float in [0,1]>,\n'
    '          "source_components": ["<component_id>", ...]'
)


def _patch_init_schema(base: str) -> str:
    """Inject the `source_components` field into the base node + edge schema."""
    base = base.replace(_NODE_SCHEMA_OLD, _NODE_SCHEMA_NEW, 1)
    base = base.replace(_EDGE_SCHEMA_OLD, _EDGE_SCHEMA_NEW, 1)
    if not (_NODE_SCHEMA_NEW in base and _EDGE_SCHEMA_NEW in base):
        raise RuntimeError(
            "kg4vd.kg.prompts.ENTITY_EXTRACTION_INIT no longer matches the "
            "expected schema markers; component-cued patching failed. "
            "Re-sync _NODE_SCHEMA_OLD / _EDGE_SCHEMA_OLD."
        )
    return base


_INIT_BASE_PATCHED = _patch_init_schema(_INIT_BASE)


# Wrap the flat `{nodes, edges}` output of the base prompt into the GPT-spec
# `ops` shape so the loop can unify init + revise outputs.
_OPS_WRAPPER = dedent(
    """
    OUTPUT WRAPPER. Wrap the nodes and edges you produce into the
    "add_nodes" and "add_edges" slots of an `ops` object. The full
    output schema is:

    {{
      "reason": "<one-sentence justification>",
      "ops": {{
        "add_nodes": [<nodes following the schema above>],
        "add_edges": [<edges following the schema above>],
        "replace_nodes": [],
        "replace_edges": [],
        "delete_nodes": [],
        "delete_edges": []
      }},
      "uncertainties": [
        {{
          "component_id": "<id>",
          "reason": "<why uncertain>",
          "requested_input": "component_crop" | "expanded_manifest" | "table_html" | "neighbor_context"
        }}
      ]
    }}

    On the first pass, only "add_nodes" / "add_edges" should be
    populated; replace / delete slots are for later revision rounds.
    """
).strip()


EXTRACTOR_INIT_PROMPT = (
    _ANNOTATED_IMAGE_BLOCK
    + "\n\n"
    + _GROUNDING_BLOCK
    + "\n\n---\n\n"
    + _INIT_BASE_PATCHED
    + "\n\n---\n\n"
    + _OPS_WRAPPER
    + "\n\nComponent manifest:\n{component_manifest}\n"
)
"""Round-0 Extractor prompt.

Required `.format()` placeholders (inherited from `ENTITY_EXTRACTION_INIT`
plus our own):
  - entity_types, visual_entity_types
  - page_id, total_pages
  - page_text_block
  - component_manifest
"""


# ---------------------------------------------------------------------------
# 2. Reflector prompt - emits a RevisionBrief, NEVER writes ops
# ---------------------------------------------------------------------------

REFLECTOR_PROMPT = dedent(
    """
    You are a COVERAGE JUDGE for a multimodal KG-extraction pipeline.
    You are NOT a graph writer. Your job is to decide whether the
    current KG already covers the page well enough, and only if it
    doesn't, to point at the specific gap(s) the next Extractor pass
    should close.

    Stopping is the correct outcome whenever the page is well-covered.
    A trivial / empty brief with `stop_recommendation: true` is a
    SUCCESS, not a failure - do NOT invent work to fill the slots.

    {annotated_image_block}

    {grounding_block}

    INPUTS YOU SEE:
      1. The annotated page image (boxes + circle labels).
      2. The component manifest (text-component OCR + bbox positions).
      3. **Component scorecards** - one card per component, showing
         its manifest content alongside the nodes and edges already
         grounded to it. The scorecards ARE the source of truth about
         what is already covered. Do NOT propose entities or relations
         that the scorecard already lists.

    DECISION PROCEDURE - follow these steps IN ORDER:

      STEP 1 - Coverage judgement (binary).
        For each non-chrome component, look at its scorecard and ask:
        "given the manifest content, do the nodes / edges already
        grounded here adequately represent it?"

        DENSITY CHECK (mandatory for ALL non-chrome components -
        textual, table, AND figure):
          - If the manifest text is an ENUMERATION - i.e. it lists
            multiple named items separated by commas / semicolons /
            "and" / newlines (typical patterns: contributor lists,
            reviewer lists, bullet lists, indexes, donor lists, tables
            of names + affiliations) - then count the items in the
            manifest text and compare to the number of nodes grounded
            here.
              * If the count is clearly mismatched (e.g. manifest lists
                ~22 names but scorecard shows 1 node), the component
                is `partially_covered` or `uncovered`, NEVER `covered`.
              * Optimistic phrases like "all contributors included"
                are PROHIBITED unless the node count actually matches.
          - For TABLES (component_id starts with `TB`): if the scorecard
            shows a `table_html` block, count the data rows (<tr>) and
            the meaningful cells (<td> entries that are not header
            labels). Compare that count to the nodes grounded to this
            table. A table with N data rows whose nodes count is 1-2
            is almost always `partially_covered`. If no `table_html`
            is shown, you must inspect the annotated image to estimate
            row count before declaring `covered`.
          - If the manifest text describes 1+ named entities (people,
            organisations, places, dates, works) and the scorecard
            shows ZERO nodes grounded here, the status is `uncovered`,
            NEVER `covered`.
          - FIGURE / IMAGE components (component_id starts with `IM`
            or `FIG`): inspect the annotated page image (cropped to
            this region) and count the visually distinct subjects -
            distinct characters, distinct objects, distinct chart
            elements, distinct sub-panels in a grid.
              * If the figure shows N≥2 distinct subjects AND the
                scorecard has fewer than N nodes grounded here, status
                is `partially_covered` (under_decomposed_visual).
              * Examples that should NOT be 1-node: a grid of
                emotional poses (each pose ≈ 1 node), a scene with
                multiple characters (each character ≈ 1 node), a
                Venn diagram with multiple subfields (each subfield
                ≈ 1 node).
              * Single-subject figures (one chart, one portrait,
                one object) are fine with 1 node.

        CONSISTENCY RULE:
          - The `notes` field MUST be consistent with `status`. If your
            note says "no nodes grounded" / "missing X" / "only Y of Z
            included", the status CANNOT be `covered`. Pick
            `partially_covered` or `uncovered` instead.

        STOPPING CRITERION:
          - You may set `stop_recommendation: true` ONLY when BOTH of:
              (i)  every non-chrome component is `covered` or clearly
                   irrelevant,
              (ii) no figure component has fewer nodes than its
                   visually distinct subject count.
          - If you stop, set `stop_recommendation: true`, leave
            `critiques`, `focus_cues`, `suggested_ops` empty, and
            write a one-line `summary` saying "page covered". STOP.
            Do not continue to Step 2.
          - If you cannot satisfy (i)+(ii), continue to Step 2.

      STEP 2 - Specific, evidence-backed gaps (only if Step 1 said NO).
        For every gap you raise, you MUST be able to point at a
        specific scorecard line as evidence. If you cannot quote the
        scorecard evidence, do NOT emit the critique. Acceptable
        evidence shapes:
          * "P4 manifest text lists names A, B, C, ... but scorecard
             shows only 2 person nodes grounded here"
          * "IM2 has zero nodes grounded here and its caption refers
             to a Venn diagram"
          * "edge X --[rel]--> Y cites P1 but P1's manifest text does
             not mention Y"

      STEP 3 - Cues + suggestions (only for the gaps from Step 2).
        - `focus_cues`: one entry per gap component (priority + reason).
        - `component_reviews`: short status per non-chrome component
          (covered / partially_covered / uncovered / needs_visual /
          irrelevant) with brief notes.
        - `critiques`: the specific issues - unsupported_relation,
          wrong_grounding, vague_visual_description, duplicate_entity,
          wrong_type, over_specific_claim, under_decomposed_visual,
          missing_entity, missing_relation.
        - `suggested_ops`: optional recommendations the next Extractor
          may accept, modify, or reject. Use the same node/edge schema
          as the Extractor (with `source_components`).

    HARD RULES:
      - Vague critiques without scorecard evidence are PROHIBITED. If
        you cannot quote which scorecard line shows the gap, drop it.
      - You CANNOT write authoritative ops. Your `suggested_ops` are
        hints, not commands.
      - Every focus_cue / critique / suggested_op MUST cite real
        component IDs from the manifest.
      - Do NOT propose to extract "P3" or "IM1" itself as an entity.

    OUTPUT - JSON only, no markdown:
    {{
      "page_id": <int>,
      "summary": "<one-paragraph snapshot of the current KG state>",
      "focus_cues": [
        {{
          "component_id": "<id>",
          "priority": "high" | "medium" | "low",
          "reason": "<why this needs another look>",
          "requested_input": "annotated_page_only" | "component_crop" | "expanded_manifest" | "table_html" | "neighbor_context"
        }}
      ],
      "component_reviews": [
        {{
          "component_id": "<id>",
          "status": "covered" | "partially_covered" | "uncovered" | "needs_visual_inspection" | "irrelevant",
          "notes": ["<short observation>", ...]
        }}
      ],
      "critiques": [
        {{
          "target_kind": "node" | "edge" | "component" | "coverage",
          "target_ref": <object>,
          "issue_type": "<one of the issue types listed above>",
          "severity": "high" | "medium" | "low",
          "comment": "<why>"
        }}
      ],
      "suggested_ops": {{
        "add_nodes": [...],
        "add_edges": [...],
        "replace_nodes": [...],
        "replace_edges": [...],
        "delete_nodes": [...],
        "delete_edges": []
      }},
      "stop_recommendation": <bool>
    }}

    EXAMPLES (illustrative; the actual page below is different).
    Both examples share a small fictional "MegaCorp founded in 1886 by
    John Smith in Chicago; Figure 1 shows the logo" page, so you can
    compare Reflector behaviour on covered vs. partially-covered KGs.

    --- EXAMPLE A - page covered → STOP (the common case) ---
    Component scorecards (excerpt):
      == T1 (title, top) ==
        manifest text: MegaCorp
        nodes grounded here (1): MegaCorp (organization, text)
        edges grounded here (0): (none)
      == P1 (paragraph, body) ==
        manifest text: Founded in 1886, MegaCorp was established by
                       John Smith in Chicago. The company specialises
                       in widgets.
        nodes grounded here (4): MegaCorp (organization, text);
                                  John Smith (person, text);
                                  1886 (date, text);
                                  Chicago (location, text)
        edges grounded here (3): MegaCorp -[founded by]-> John Smith;
                                  MegaCorp -[founded in]-> 1886;
                                  MegaCorp -[located in]-> Chicago
      == IM1 (image, right) ==
        manifest: <image - see annotated page>
        nodes grounded here (1): MegaCorp logo (visual_object, visual)
        edges grounded here (1): MegaCorp logo -[depicts]-> MegaCorp

    A good RevisionBrief - page is covered, so stop cleanly:
    {{
      "page_id": 1,
      "summary": "Page covered: title, body paragraph, and logo all have appropriate grounded nodes and edges.",
      "focus_cues": [],
      "component_reviews": [
        {{"component_id":"T1","status":"covered","notes":["title entity grounded"]}},
        {{"component_id":"P1","status":"covered","notes":["all 4 entities + 3 stated relations present"]}},
        {{"component_id":"IM1","status":"covered","notes":["logo + depicts edge present"]}}
      ],
      "critiques": [],
      "suggested_ops": {{
        "add_nodes": [], "add_edges": [],
        "replace_nodes": [], "replace_edges": [],
        "delete_nodes": [], "delete_edges": []
      }},
      "stop_recommendation": true
    }}
    Takeaway: when scorecards show every component covered, emit an
    empty brief with `stop_recommendation: true`. Do NOT invent
    low-confidence critiques to fill the slots.

    --- EXAMPLE B - real, evidence-backed gap → focused brief ---
    Same page, but the current KG is missing the logo and the founder-
    location edge:
      == T1 (title, top) ==
        manifest text: MegaCorp
        nodes grounded here (1): MegaCorp (organization, text)
      == P1 (paragraph, body) ==
        manifest text: Founded in 1886, MegaCorp was established by
                       John Smith in Chicago. The company specialises
                       in widgets.
        nodes grounded here (4): MegaCorp; John Smith; 1886; Chicago
        edges grounded here (2): MegaCorp -[founded by]-> John Smith;
                                  MegaCorp -[founded in]-> 1886
      == IM1 (image, right) ==
        manifest: <image - see annotated page>
        nodes grounded here (0): (none)
        edges grounded here (0): (none)

    A good RevisionBrief - each gap cites a specific scorecard line:
    {{
      "page_id": 1,
      "summary": "P1 is missing the founder-location relation; IM1 has zero nodes despite the annotated page showing a visible logo.",
      "focus_cues": [
        {{"component_id":"IM1","priority":"high",
          "reason":"Scorecard shows 0 nodes grounded to IM1 but the annotated page has a visible image at this position.",
          "requested_input":"annotated_page_only"}},
        {{"component_id":"P1","priority":"medium",
          "reason":"Scorecard shows John Smith and Chicago both grounded to P1, but no edge connects them despite P1 stating 'John Smith ... in Chicago'.",
          "requested_input":"annotated_page_only"}}
      ],
      "component_reviews": [
        {{"component_id":"T1","status":"covered","notes":[]}},
        {{"component_id":"P1","status":"partially_covered",
          "notes":["John Smith -[founded MegaCorp in]-> Chicago edge implied by text but missing"]}},
        {{"component_id":"IM1","status":"uncovered",
          "notes":["no visual entity yet"]}}
      ],
      "critiques": [
        {{"target_kind":"coverage","target_ref":{{"component_id":"IM1"}},
          "issue_type":"missing_entity","severity":"high",
          "comment":"IM1 scorecard has zero nodes; the annotated page shows a logo at this position."}},
        {{"target_kind":"coverage","target_ref":{{"component_id":"P1"}},
          "issue_type":"missing_relation","severity":"medium",
          "comment":"P1 text says 'John Smith ... in Chicago' but no edge links them in the scorecard."}}
      ],
      "suggested_ops": {{
        "add_nodes": [
          {{"name":"MegaCorp logo","entity_type":"visual_object","modality":"visual",
            "visual_type":"visual_object",
            "description":"Logo of MegaCorp shown in Figure 1.",
            "visual_description":"Blue gear surrounded by three concentric rings, with 'MegaCorp' wordmark below.",
            "source_components":["IM1"]}}
        ],
        "add_edges": [
          {{"src":"MegaCorp logo","tgt":"MegaCorp","relation":"depicts",
            "description":"Figure 1 is the visual emblem of MegaCorp.","confidence":0.95,
            "source_components":["IM1"]}},
          {{"src":"John Smith","tgt":"Chicago","relation":"founded MegaCorp in",
            "description":"P1 states John Smith established MegaCorp in Chicago.","confidence":0.85,
            "source_components":["P1"]}}
        ],
        "replace_nodes": [], "replace_edges": [],
        "delete_nodes": [], "delete_edges": []
      }},
      "stop_recommendation": false
    }}
    Takeaways: every critique cites a SPECIFIC scorecard observation
    ("0 nodes grounded to IM1", "P1 text says X but no edge in scorecard").
    Every suggested op carries `source_components` referencing real
    component IDs.

    --- EXAMPLE C - enumeration density mismatch (DO NOT call this "covered") ---
    A page where component P4 is a contributor list. Scorecard:
      == P4 (paragraph, body) ==
        manifest text: Chris Bingley, UCLA; Celeste Chamberland,
                       Roosevelt University; Scott Corbett, Ventura
                       College; Rick Gianni, Grand Canyon University;
                       Jennifer Lawrence, Tarrant County College;
                       Jamie McCandless, Kennesaw State University;
                       Cristina Mehrtens, University of Massachusetts
                       Dartmouth; Anthony Miller, Hanover College;
                       ... (22 names total)
        nodes grounded here (1): Contributing Authors (group, text)
        edges grounded here (0): (none)
      == T4 (title, body) ==
        manifest text: Reviewers
        nodes grounded here (0): (none)
        edges grounded here (0): (none)

    A WRONG verdict (DO NOT do this):
      "component_reviews": [
        {{"component_id":"P4","status":"covered","notes":["all contributors included"]}},
        {{"component_id":"T4","status":"covered","notes":["title entity grounded"]}}
      ]
      "stop_recommendation": true
    Why this is wrong: P4 manifest enumerates 22 names but scorecard
    shows 1 group node, not 22 person nodes - the "all contributors
    included" claim contradicts the count. T4's note literally says
    "title entity grounded" but nodes grounded here = 0, a direct
    contradiction.

    The RIGHT verdict - count the items and report the gap honestly:
    {{
      "page_id": 5,
      "summary": "P4 lists 22 contributors but only 1 group node is grounded; T4 title has zero nodes despite being a section header.",
      "focus_cues": [
        {{"component_id":"P4","priority":"high",
          "reason":"Scorecard shows 1 node grounded to P4 but the manifest text enumerates 22 named contributors with their affiliations. Need 22 person nodes plus their affiliated organization nodes.",
          "requested_input":"annotated_page_only"}},
        {{"component_id":"T4","priority":"low",
          "reason":"Scorecard shows 0 nodes grounded to T4 but the title is 'Reviewers', which is a meaningful section label.",
          "requested_input":"annotated_page_only"}}
      ],
      "component_reviews": [
        {{"component_id":"P4","status":"partially_covered",
          "notes":["manifest lists 22 named contributors but only 1 group node is grounded - need per-person entities"]}},
        {{"component_id":"T4","status":"uncovered",
          "notes":["title text 'Reviewers' but 0 nodes grounded here"]}}
      ],
      "critiques": [
        {{"target_kind":"coverage","target_ref":{{"component_id":"P4"}},
          "issue_type":"under_decomposed_visual","severity":"high",
          "comment":"P4 enumerates 22 contributors but scorecard has 1 collapsed group node; each named contributor should be its own person node, with affiliation edges to their organizations."}}
      ],
      "suggested_ops": {{
        "add_nodes": [
          {{"name":"Chris Bingley","entity_type":"person","modality":"text",
            "description":"Contributing author, affiliated with UCLA.",
            "source_components":["P4"]}},
          {{"name":"UCLA","entity_type":"organization","modality":"text",
            "description":"University of California, Los Angeles; affiliation of Chris Bingley.",
            "source_components":["P4"]}}
        ],
        "add_edges": [
          {{"src":"Chris Bingley","tgt":"UCLA","relation":"affiliated with",
            "description":"P4 lists Chris Bingley with UCLA as affiliation.","confidence":0.95,
            "source_components":["P4"]}}
        ],
        "replace_nodes": [], "replace_edges": [],
        "delete_nodes": [], "delete_edges": []
      }},
      "stop_recommendation": false
    }}
    Takeaway: enumerations REQUIRE that the node count be in the same
    ballpark as the item count. "1 group node" is not coverage of 22
    named individuals. Pattern-match the scorecard count, not the
    structure.

    --- end of examples ---

    Component manifest:
    {component_manifest}

    Current KG state (grouped by source component):
    {current_kg_block}

    Page number: {page_id} of {total_pages}
    {page_text_block}
    """
).strip()


# ---------------------------------------------------------------------------
# 3. Extractor - REVISE prompt (round 2, 4, ...)
# ---------------------------------------------------------------------------

EXTRACTOR_REVISE_PROMPT = dedent(
    """
    You are the graph writer, revising a multimodal KG for this page.

    {annotated_image_block}

    {grounding_block}

    INPUTS YOU SEE:
      1. The annotated page image. Components rendered in MUTED GREY
         with hollow circle labels are SATURATED - their content is
         already extracted. Components in BLUE / RED / GREEN with
         filled circles are still ACTIVE.
      2. The component manifest. Saturated entries show
         `[saturated - N nodes already grounded here; see scorecard.
         Do NOT re-extract; use these entities as edge endpoints
         only.]` in place of the raw text. Do NOT propose new entities
         for these components.
      3. **Component scorecards** - one card per component, showing
         what is already grounded to it (nodes + edges). Do NOT
         re-emit operations that the scorecard already lists.
         Saturated entities are still listed here so you can build
         cross-component edges referencing them.
      4. A `RevisionBrief` from the Reflector listing focus_cues,
         critiques, and SUGGESTED operations.

    YOUR JOB:
      - Decide which Reflector suggestions are valid and emit them as
        AUTHORITATIVE ops. You may accept, modify, or reject any
        suggestion.
      - Independently, address any high-priority focus_cue or
        high-severity critique by emitting your own ops as needed.
      - Every node / edge you emit MUST carry `source_components`.

    OUTPUT - JSON only:
    {{
      "reason": "<one-sentence summary of changes>",
      "ops": {{
        "add_nodes": [...],
        "add_edges": [...],
        "replace_nodes": [...],
        "replace_edges": [...],
        "delete_nodes": ["<canonical name>", ...],
        "delete_edges": [{{ "src": "...", "tgt": "...", "relation": "..." }}]
      }},
      "uncertainties": [
        {{
          "component_id": "<id>",
          "reason": "<why uncertain>",
          "requested_input": "<...>"
        }}
      ],
      "rejected_reflector_suggestions": [
        {{
          "ref": "<short reference to the rejected suggestion>",
          "reason": "<why rejected>"
        }}
      ]
    }}

    HARD RULES:
      - Use the same node / edge schema as the init pass (with
        `source_components`).
      - Do NOT extract "P3" / "IM1" as entities.
      - `delete_nodes` and `delete_edges` cite canonical-name / (src,
        tgt, relation) of the existing target.
      - `replace_nodes` is ONLY for upgrading auto-materialised stubs
        (`entity_type == "entity"`). Specifically:
            * The `name` field MUST match an existing stub (the
              scorecard marks it `[STUB - upgrade via replace_nodes]`,
              and the brief lists the exact set in "Stubs to upgrade").
            * The new `entity_type` MUST be a specific type from the
              recipe vocab (organization, person, location, date,
              scientific_concept, etc.) - NEVER `entity`.
            * Do NOT replace a node that is already correctly typed.
              The controller will drop such ops.

    STUB UPGRADE - IMPORTANT.
      The pipeline auto-materialises stub nodes for orphan edge
      endpoints (entity_type="entity", description starting with
      "Auto-materialised stub..."). When you see such a stub in the
      scorecard AND you can determine the real semantic type from the
      cited component's text (or its image), emit a `replace_nodes`
      entry to upgrade it in place:

        - Use the SAME `name` as the stub (case-insensitive match);
          `apply_ops` will replace by canonical name.
        - Set `entity_type` to a more specific type from the recipe
          vocab (e.g. organization, person, location, date,
          scientific_concept).
        - Write a proper `description` based on the cited text.
        - Keep `source_components` consistent with the stub's grounding.

      Examples of stub names that should be upgraded each round:
        - "UCLA" (cited P4) → organization
        - "2015"  (cited P3) → date
        - "Roosevelt University" (cited P4) → organization
      Failing to upgrade leaves the KG with weakly-typed entities.

    EXAMPLE (illustrative; the actual page below is different).
    Suppose the scorecard shows IM1 with 0 nodes and P1 missing the
    John-Smith-in-Chicago edge, and the Reflector's RevisionBrief is:

      Summary: P1 missing founder-location relation; IM1 uncovered.
      Focus cues:
        - IM1 (priority=high): 0 nodes grounded but logo visible.
        - P1 (priority=medium): no edge between existing John Smith
          and Chicago even though P1 states they are linked.
      Suggested ops (recommendations only, 3 total):
        add_nodes:
          - {{"name":"MegaCorp logo","entity_type":"visual_object",
              "modality":"visual","visual_type":"visual_object",
              "visual_description":"Blue gear with three concentric rings.",
              "source_components":["IM1"]}}
        add_edges:
          - {{"src":"MegaCorp logo","tgt":"MegaCorp","relation":"depicts",
              "source_components":["IM1"]}}
          - {{"src":"John Smith","tgt":"Chicago","relation":"founded MegaCorp in",
              "source_components":["P1"]}}

    A good authoritative patch - accept the two grounded ops, but
    UPGRADE the visual node's description after looking at the
    annotated page (which actually shows a 4-ring logo, not 3):
    {{
      "reason": "Accept the missing logo and founder-location ops; tighten the visual_description after inspecting the annotated page.",
      "ops": {{
        "add_nodes": [
          {{"name":"MegaCorp logo","entity_type":"visual_object","modality":"visual",
            "visual_type":"visual_object",
            "description":"Logo of MegaCorp shown in Figure 1.",
            "visual_description":"Blue gear ~3cm across surrounded by four concentric rings of decreasing thickness; sans-serif 'MegaCorp' wordmark in dark blue centred below the gear.",
            "source_components":["IM1"]}}
        ],
        "add_edges": [
          {{"src":"MegaCorp logo","tgt":"MegaCorp","relation":"depicts",
            "description":"Figure 1 is the visual emblem of MegaCorp.",
            "confidence":0.95,
            "source_components":["IM1"]}},
          {{"src":"John Smith","tgt":"Chicago","relation":"founded MegaCorp in",
            "description":"P1 states John Smith established MegaCorp in Chicago.",
            "confidence":0.85,
            "source_components":["P1"]}}
        ],
        "replace_nodes": [], "replace_edges": [],
        "delete_nodes": [], "delete_edges": []
      }},
      "uncertainties": [],
      "rejected_reflector_suggestions": []
    }}
    Takeaways: every emitted node / edge carries `source_components`
    citing a real component ID. The Reflector's `visual_description`
    was upgraded because the image showed more detail - accept,
    modify, or reject suggestions; you are the authoritative writer.

    --- EXAMPLE 2 - replace_nodes to upgrade auto-materialised stubs ---
    Suppose the scorecard contains stubs the validator created for
    edge endpoints the previous round forgot to emit:

      == P4 (paragraph, body) ==
        manifest text: Chris Bingley, UCLA; Celeste Chamberland,
                       Roosevelt University; ...
        nodes grounded here (3):
          - Chris Bingley (person, text)
          - UCLA (entity, text) - Auto-materialised stub
          - Roosevelt University (entity, text) - Auto-materialised stub
        edges grounded here (2):
          - Chris Bingley -[affiliated with]-> UCLA
          - Celeste Chamberland -[affiliated with]-> Roosevelt University

    The Reflector's brief flags the stubs as weakly typed:
      Critiques:
        - [medium] wrong_type: "UCLA" / "Roosevelt University" are
          stub-typed; manifest text shows they are universities.

    A good authoritative patch - upgrade the stubs via `replace_nodes`,
    matched by canonical name:
    {{
      "reason": "Upgrade two auto-materialised stubs to organization entities based on P4 text.",
      "ops": {{
        "add_nodes": [],
        "add_edges": [],
        "replace_nodes": [
          {{"name":"UCLA","entity_type":"organization","modality":"text",
            "description":"University of California, Los Angeles; affiliation of contributor Chris Bingley as listed in P4.",
            "source_components":["P4"]}},
          {{"name":"Roosevelt University","entity_type":"organization","modality":"text",
            "description":"Roosevelt University, affiliation of contributor Celeste Chamberland as listed in P4.",
            "source_components":["P4"]}}
        ],
        "replace_edges": [],
        "delete_nodes": [],
        "delete_edges": []
      }},
      "uncertainties": [],
      "rejected_reflector_suggestions": []
    }}
    Takeaways: the `name` matches the existing stub exactly so
    `apply_ops` replaces in place; `entity_type` upgrades from "entity"
    to a specific recipe-vocab type; `source_components` stays the
    same as the stub's grounding. The edges connecting to these
    upgraded nodes do NOT need to be re-emitted - they already exist.

    Component manifest:
    {component_manifest}

    Current KG state (grouped by source component):
    {current_kg_block}

    Reflector's RevisionBrief:
    {revision_brief_block}

    Page number: {page_id} of {total_pages}
    {page_text_block}
    """
).strip()


# Plug the shared blocks into the templates that reference them.
# (The `{annotated_image_block}` and `{grounding_block}` placeholders
# are statically substituted here; remaining `{...}` placeholders are
# filled at .format() time by the extractor.)
REFLECTOR_PROMPT = REFLECTOR_PROMPT.replace(
    "{annotated_image_block}", _ANNOTATED_IMAGE_BLOCK
).replace("{grounding_block}", _GROUNDING_BLOCK)

EXTRACTOR_REVISE_PROMPT = EXTRACTOR_REVISE_PROMPT.replace(
    "{annotated_image_block}", _ANNOTATED_IMAGE_BLOCK
).replace("{grounding_block}", _GROUNDING_BLOCK)


__all__ = [
    "PROMPTS_VERSION_COMPONENT_CUED",
    "EXTRACTOR_INIT_PROMPT",
    "REFLECTOR_PROMPT",
    "EXTRACTOR_REVISE_PROMPT",
]
