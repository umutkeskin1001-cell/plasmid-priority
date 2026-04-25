"""Local retrieval helpers for jury-facing RAG summaries."""


import re
from dataclasses import dataclass
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]{2,}")


def _tokenize(text: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(text)}


@dataclass(frozen=True)
class RagChunk:
    path: str
    title: str
    text: str
    tokens: frozenset[str]


def load_rag_chunks(paths: list[Path]) -> list[RagChunk]:
    """Load text documents as retrieval chunks."""
    chunks: list[RagChunk] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        title = path.name
        first_line = text.splitlines()[0].strip() if text.splitlines() else ""
        if first_line.startswith("#"):
            title = first_line.lstrip("#").strip() or title
        chunks.append(
            RagChunk(
                path=str(path),
                title=title,
                text=text,
                tokens=frozenset(_tokenize(text)),
            ),
        )
    return chunks


def retrieve_rag(
    chunks: list[RagChunk],
    query: str,
    *,
    top_k: int = 5,
) -> list[dict[str, object]]:
    """Return top matching chunks using token overlap scoring."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []
    scored: list[tuple[float, RagChunk]] = []
    for chunk in chunks:
        overlap = len(query_tokens.intersection(chunk.tokens))
        if overlap == 0:
            continue
        precision = overlap / max(1, len(chunk.tokens))
        recall = overlap / len(query_tokens)
        score = 0.7 * recall + 0.3 * precision
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    results: list[dict[str, object]] = []
    for score, chunk in scored[: max(1, int(top_k))]:
        snippet = chunk.text[:400].replace("\n", " ").strip()
        results.append(
            {
                "path": chunk.path,
                "title": chunk.title,
                "score": round(score, 6),
                "snippet": snippet,
            },
        )
    return results
