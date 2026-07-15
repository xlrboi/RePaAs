"""All the constants are stored in this file.
"""

from __future__ import annotations

# ── Document chunking ────────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Vector store ──────────────────────────────────────────────────────────────
EMBEDDING_DIM = 1536  # text-embedding-3-small
DEFAULT_RETRIEVAL_K = 4
MIN_RETRIEVAL_K = 1
MAX_RETRIEVAL_K = 10

# ── Web search ────────────────────────────────────────────────────────────────
DEFAULT_WEB_RESULTS = 3
MIN_WEB_RESULTS = 1
MAX_WEB_RESULTS = 10
BTW_WEB_RESULTS = 3
CLAIM_VERIFICATION_WEB_RESULTS = 5

# ── Retrieval agent loop control ────────────────────────────────────────────
MAX_RETRIEVAL_ATTEMPTS = 3
MAX_QUERY_REWRITES = 1

# ── Misc ──────────────────────────────────────────────────────────────────────
SESSION_NAME_SOURCE_CHARS = 500
CLAIM_SNIPPET_CHARS = 200
DOC_PREVIEW_CHARS = 300
