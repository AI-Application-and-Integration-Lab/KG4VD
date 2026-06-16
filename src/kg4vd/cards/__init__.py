"""Evidence Card builders - turn Pages + KG into typed cards.

The construction pipeline emits only the card types the downstream query
side actually retrieves: ``page`` (anchor retrieval) and ``entity`` /
``relation`` (KG-bridge embeddings). The ``text_chunk`` / ``page_summary`` /
``subgraph`` / ``*_summary`` card types are not built - the query side
never retrieved them.
"""

from kg4vd.cards.builders import (
    build_entity_cards,
    build_page_cards,
    build_relation_cards,
)

__all__ = [
    "build_entity_cards",
    "build_page_cards",
    "build_relation_cards",
]
