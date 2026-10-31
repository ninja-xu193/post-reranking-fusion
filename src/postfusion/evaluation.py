"""Linear-gain nDCG, total-relevant AP denominators and explicit ranked IDs."""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def evaluate(query, order):
    if len(order)!=len(query.doc_ids) or set(order)!=set(query.doc_ids):
        raise ValueError("Order must be a permutation of all candidate IDs")
    grades=np.array([query.qrels.get(d,0) for d in order],dtype=float)
    ideal=np.array(sorted(query.qrels.values(),reverse=True)[:10],dtype=float)
    discounts=1/np.log2(np.arange(2,len(ideal)+2))
    idcg=float(ideal@discounts)
    top=grades[:10]; binary=(top>0).astype(float)
    relevant=sum(v>0 for v in query.qrels.values())
    dcg=float(top@(1/np.log2(np.arange(2,len(top)+2))))
    ap=float(np.sum(binary*np.cumsum(binary)/np.arange(1,len(binary)+1)))
    return {"ndcg@10":dcg/idcg if idcg else 0.0,
            "map@10":ap/relevant if relevant else 0.0,
            "recall@5":float(np.sum(grades[:5]>0))/relevant if relevant else 0.0}


def macro_average(queries, values):
    queries,values=list(queries),list(values)
    if not queries or len(queries)!=len(values) or not np.isfinite(values).all():
        raise ValueError("Aligned, nonempty finite query values are required")
    domains=defaultdict(list)
    for q,value in zip(queries,values):
        domains[q.domain].append(value)
    return float(np.mean([np.mean(v) for v in domains.values()]))


def tie_gain_ratio(direct, tie, oracle):
    if not np.isfinite([direct,tie,oracle]).all():
        raise ValueError("Metric values must be finite")
    denominator=oracle-direct
    return (tie-direct)/denominator if denominator>0 else None
