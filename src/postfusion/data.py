"""Strict JSONL input with full-query judgments kept separate from candidates."""
from __future__ import annotations
from dataclasses import dataclass
import json
from types import MappingProxyType
from pathlib import Path
from typing import Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class Query:
    query_id: str
    domain: str
    doc_ids: Tuple[str, ...]
    bm25: Tuple[float, ...]
    reranker: Tuple[float, ...]
    qrels: Mapping[str, float]

    def __post_init__(self):
        if not isinstance(self.query_id,str) or not isinstance(self.domain,str) or not self.query_id or not self.domain:
            raise ValueError("query_id and domain must be nonempty")
        object.__setattr__(self,"doc_ids",tuple(self.doc_ids))
        object.__setattr__(self,"bm25",tuple(self.bm25))
        object.__setattr__(self,"reranker",tuple(self.reranker))
        object.__setattr__(self,"qrels",MappingProxyType(dict(self.qrels)))
        if not self.doc_ids or len(set(self.doc_ids)) != len(self.doc_ids):
            raise ValueError("Candidates must be nonempty with unique document IDs")
        if any(not isinstance(d, str) or not d for d in self.doc_ids):
            raise ValueError("Document IDs must be nonempty strings")
        if len(self.doc_ids) != len(self.bm25) or len(self.doc_ids) != len(self.reranker):
            raise ValueError("Candidate and score lengths differ")
        for name in ("bm25", "reranker"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.ndim != 1 or not np.isfinite(values).all():
                raise ValueError("Scores must be finite one-dimensional vectors")
            object.__setattr__(self, name, tuple(float(v) for v in values))
        if any(not np.isfinite(v) or v < 0 for v in self.qrels.values()):
            raise ValueError("Judgment grades must be finite and nonnegative")

    @property
    def key(self):
        return self.domain, self.query_id

    @property
    def labels(self):
        return np.array([self.qrels.get(d, 0) > 0 for d in self.doc_ids], dtype=float)


def load_queries(path):
    queries = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                docs = row["candidates"]
                query = Query(row["query_id"], row["domain"],
                              tuple(d["doc_id"] for d in docs),
                              tuple(float(d["bm25"]) for d in docs),
                              tuple(float(d["reranker"]) for d in docs),
                              {str(k): float(v) for k, v in row["qrels"].items()})
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid query record on line {number}: {exc}") from exc
            queries.append(query)
    if not queries:
        raise ValueError("Input contains no queries")
    if len({q.key for q in queries}) != len(queries):
        raise ValueError("Duplicate domain/query identity")
    return queries
