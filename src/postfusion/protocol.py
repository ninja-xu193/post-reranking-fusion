"""Source-only holdouts and query-local score permutations."""
from dataclasses import replace
import hashlib
import numpy as np

from .learning import fit_listfuse, fit_logistic, select_alpha
from .ranking import rank
from .evaluation import evaluate


def permute_bm25(query, seed):
    key=f"bm25-permutation\0{seed}\0{query.domain}\0{query.query_id}"
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"big"))
    # Canonicalize by ID so record ordering cannot change the permutation.
    ids=sorted(range(len(query.doc_ids)),key=lambda i:query.doc_ids[i])
    values=np.asarray(query.bm25)[ids]
    shuffled=rng.permutation(values)
    b=np.empty_like(values)
    b[ids]=shuffled
    return replace(query,bm25=tuple(b))


def leave_one_domain_out(queries, method="listfuse", domain_equal=False,
                          grid_step=None, seeds=range(20)):
    """Return held-out query metrics; fitting and alpha choice see source domains only."""
    queries,seeds=list(queries),tuple(seeds)
    domains=sorted({q.domain for q in queries})
    if len(domains)<2 or not seeds:
        raise ValueError("At least two domains and one seed are required")
    rows=[]
    for target in domains:
        train=[q for q in queries if q.domain!=target]
        test=[q for q in queries if q.domain==target]
        model=None; alpha=.25
        if method in ("listfuse","ltb"):
            model=fit_listfuse(train,domain_equal,grid_step,tie_restricted=method=="ltb")
        elif method=="logistic":
            if domain_equal:
                raise ValueError("domain-equal is available for listwise fits only")
            model=fit_logistic(train,grid_step)
        elif method=="convex":
            alpha=select_alpha(train,seeds=seeds,grid_step=grid_step)
        for q in test:
            repeats=[evaluate(q,rank(q,method,seed,model,alpha,grid_step)) for seed in seeds]
            rows.append({"domain":target,"query_id":q.query_id,
                         "metrics":{k:float(np.mean([v[k] for v in repeats])) for k in repeats[0]}})
    return rows
