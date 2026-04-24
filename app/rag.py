from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    id: str
    title: str
    text: str
    tags: list[str]


def _corpus_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "corpus.jsonl"


@lru_cache(maxsize=1)
def load_corpus() -> list[Chunk]:
    path = _corpus_path()
    if not path.exists():
        return []
    chunks: list[Chunk] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        chunks.append(
            Chunk(
                id=row["id"],
                title=row["title"],
                text=row["text"],
                tags=list(row.get("tags", [])),
            )
        )
    return chunks


_TOKEN = re.compile(r"[a-z0-9°]+", re.I)


def _tokens(s: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(s)}


def retrieve(query: str, k: int = 4) -> list[Chunk]:
    """Very simple overlap ranker — replace with embeddings for research scale."""
    q = _tokens(query)
    if not q:
        return load_corpus()[:k]
    scored: list[tuple[int, Chunk]] = []
    for c in load_corpus():
        bag = _tokens(c.title + " " + c.text + " " + " ".join(c.tags))
        score = len(q & bag)
        scored.append((score, c))
    scored.sort(key=lambda x: (-x[0], x[1].id))
    return [c for s, c in scored if s > 0][:k] or load_corpus()[:k]


def retrieve_merged(queries: list[str], *, k_per_query: int = 3, max_chunks: int = 10) -> list[Chunk]:
    """Run retrieve for several queries and merge by chunk id (internal multi-step tool use)."""
    seen: set[str] = set()
    out: list[Chunk] = []
    for raw in queries:
        q = (raw or "").strip()
        if not q:
            continue
        for c in retrieve(q, k=k_per_query):
            if c.id not in seen:
                seen.add(c.id)
                out.append(c)
                if len(out) >= max_chunks:
                    logger.debug("retrieve_merged -> %d chunks (cap)", len(out))
                    return out
    logger.debug("retrieve_merged queries=%d -> %d chunks", len(queries), len(out))
    return out
