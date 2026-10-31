"""Ranking keys are explicit; relevance labels enter only the oracle rule."""
from __future__ import annotations
import hashlib
import numpy as np
from scipy.stats import rankdata

FEATURES = ("bm25_minmax", "reranker_minmax", "bm25_reciprocal_rank",
            "reranker_reciprocal_rank", "bm25_relative_rank", "reranker_relative_rank",
            "signed_rank_difference", "absolute_rank_difference", "score_product",
            "score_difference", "rrf_minmax")


def normalize(values):
    x = np.asarray(values, dtype=float)
    span = np.ptp(x)
    return (x - x.min()) / span if span else np.zeros_like(x)


def restore_grid(values, step=None, tolerance=1e-12):
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Scores must be finite")
    if step is None:
        return x.copy()
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Grid step must be positive and finite")
    restored = np.rint(x / step) * step
    if np.max(np.abs(x - restored)) > tolerance:
        raise ValueError("Scores are not within tolerance of the requested grid")
    return restored


def features(bm25, reranker, rrf_constant=60):
    b, s = np.asarray(bm25, dtype=float), np.asarray(reranker, dtype=float)
    if b.ndim != 1 or b.size == 0 or b.shape != s.shape:
        raise ValueError("Features require equally sized nonempty score vectors")
    if not np.isfinite(b).all() or not np.isfinite(s).all():
        raise ValueError("Scores must be finite")
    if not np.isfinite(rrf_constant) or rrf_constant < 0:
        raise ValueError("RRF constant must be nonnegative and finite")
    bn, sn = normalize(b), normalize(s)
    rb, rs = rankdata(-b, method="average"), rankdata(-s, method="average")
    k = len(b)
    return np.column_stack((bn, sn, 1/rb, 1/rs, rb/k, rs/k,
                            (rb-rs)/k, abs(rb-rs)/k, bn*sn, bn-sn,
                            normalize(1/(rrf_constant+rb)+1/(rrf_constant+rs))))


def priority(query, doc_id, seed):
    # Per-document keys are independent of candidate input order and relevance.
    key = "\0".join((str(seed), query.domain, query.query_id, doc_id))
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest(), "big")


def rank(query, method="T", seed=0, model=None, alpha=0.25,
         grid_step=None, rrf_constant=60, restrict_ties=False):
    if model is not None:
        if grid_step is not None and grid_step != model.grid_step:
            raise ValueError("Evaluation grid differs from the model's training policy")
        grid_step = model.grid_step
        rrf_constant = model.rrf_constant
    s = restore_grid(query.reranker, grid_step)
    b = np.asarray(query.bm25)
    secondary = None
    if method == "D":
        primary = s
    elif method == "T":
        primary, secondary = s, b
    elif method == "bm25":
        primary = b
    elif method == "rrf":
        if not np.isfinite(rrf_constant) or rrf_constant < 0:
            raise ValueError("Invalid RRF constant")
        primary = 1/(rrf_constant+rankdata(-b)) + 1/(rrf_constant+rankdata(-s))
    elif method == "convex":
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must lie in [0, 1]")
        primary = (1-alpha)*normalize(s) + alpha*normalize(b)
    elif method in ("listfuse", "logistic", "ltb"):
        if model is None or model.kind != {"listfuse":"listwise", "ltb":"tie_listwise", "logistic":"logistic"}[method]:
            raise ValueError("A matching fitted model is required")
        primary = model.scores(query)
        if restrict_ties or method == "ltb":
            primary, secondary = s, primary
    elif method == "oracle":
        primary = s
        secondary = np.array([query.qrels.get(d, 0) for d in query.doc_ids])
    else:
        raise ValueError(f"Unknown ranking method: {method}")
    if restrict_ties and method not in ("listfuse", "logistic", "ltb"):
        raise ValueError("restrict_ties applies only to a fitted score")
    keys = [(float(primary[i]), float(secondary[i]) if secondary is not None else 0,
             priority(query, d, seed), d) for i, d in enumerate(query.doc_ids)]
    return [query.doc_ids[i] for i in sorted(range(len(keys)), key=keys.__getitem__, reverse=True)]
