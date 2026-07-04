"""BM25 sparse vectors — tokenize client-side, send TF counts, Qdrant scores.

Why this exists: Qwen3-Embedding produces dense vectors only. To keep the
hybrid (dense + sparse / BM25) retrieval shape, we need a sparse leg. The
cleanest way without adding another model is to use Qdrant's native sparse
vectors with `Modifier.IDF` — we ship token-frequency counts as a sparse
vector, Qdrant computes IDF across the collection and scores with TF * IDF
(which is the heart of BM25).

Token IDs are stable hashes of lowercased token strings. Collisions are rare
and symmetric (the same hash function is used for both indexing and querying),
so retrieval behavior is correct even when they happen.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter

# Match word characters across unicode (so "12,456" → ["12", "456"], "FY2023" stays "fy2023").
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# 32-bit positive int space — comfortably fits Qdrant's sparse vector indices.
_HASH_MOD = 2**31


def tokenize(text: str) -> list[str]:
    """Lowercase + word-split + drop 1-char tokens.

    Kept intentionally simple. If you need stopwords or stemming, layer them
    here — but for sustainability reports, BM25's IDF naturally discounts
    common words anyway.
    """
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 2]


def _token_id(token: str) -> int:
    """Stable hash → positive int. md5 to avoid Python's salted hash()."""
    h = hashlib.md5(token.encode("utf-8")).digest()  # noqa: S324
    return int.from_bytes(h[:4], "big") % _HASH_MOD


def to_sparse(text: str) -> dict[int, float]:
    """Build a {token_id: term_frequency} sparse vector ready for Qdrant."""
    tokens = tokenize(text)
    if not tokens:
        return {}
    counts = Counter(tokens)
    return {_token_id(tok): float(cnt) for tok, cnt in counts.items()}
