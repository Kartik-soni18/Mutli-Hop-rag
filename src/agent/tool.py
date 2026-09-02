RAG_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve_documents",
        "description": (
            "Choose metadata filters for searching the private collection. "
            "Prefer source filters when publishers are named. Leave fields "
            "empty unless the question clearly supplies that metadata."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": ["string", "null"]},
                "title_text": {"type": ["string", "null"]},
                "authors": {"type": "array", "items": {"type": "string"}},
                "sources": {"type": "array", "items": {"type": "string"}},
                "published_from": {
                    "type": ["string", "null"],
                    "description": "ISO-8601 datetime",
                },
                "published_to": {
                    "type": ["string", "null"],
                    "description": "ISO-8601 datetime",
                },
                "url": {"type": ["string", "null"]},
                "url_prefix": {"type": ["string", "null"]},
            },
            "additionalProperties": False,
        },
    },
}
