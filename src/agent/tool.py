DATE_BOUND = {
    "type": ["string", "null"],
    "pattern": r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
    "format": "date",
    "description": (
        "Inclusive publication date in YYYY-MM-DD format only. "
        "Use the same date for both bounds to search one day."
    ),
}

BRANCH_PROPERTIES = {
    "retrieval_query": {
        "type": "string",
        "minLength": 1,
        "pattern": r"\S",
        "description": "A focused query for the supporting evidence in this branch.",
    },
    "source": {
        "type": ["string", "null"],
        "pattern": r"\S",
        "description": "One publisher for this branch, or null for any source.",
    },
    "authors": {"type": "array", "items": {"type": "string"}},
    "published_from": dict(DATE_BOUND),
    "published_to": dict(DATE_BOUND),
    "url": {"type": ["string", "null"]},
    "url_prefix": {"type": ["string", "null"]},
}

RAG_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve_documents",
        "description": (
            "Plan independent retrieval branches for the supporting evidence. "
            "Give each branch its own query and applicable metadata filters. "
            "Use separate branches for named publishers. Omit unsupported filters. "
            "Publication bounds are inclusive calendar dates only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "branches": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": BRANCH_PROPERTIES,
                        "required": ["retrieval_query"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["branches"],
            "additionalProperties": False,
        },
    },
}
