"""Paired saved-prediction uncertainty; no implicit refitting or family selection."""
import numpy as np


def holm(p_values):
    p=np.asarray(p_values,dtype=float)
    if p.ndim!=1 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("p-values must be a finite vector in [0, 1]")
    indices=np.argsort(p,kind="stable")
    out=np.empty_like(p); running=0.0
    for i,index in enumerate(indices):
        running=max(running,(len(p)-i)*p[index])
        out[index]=min(1.0,running)
    return out


def paired_bootstrap(left, right, repeats=20000, seed=0, hierarchical=True):
    """Inputs map (domain, query_id) to a seed-averaged metric on a 0--1 scale."""
    if not left or set(left)!=set(right):
        raise ValueError("Paired query identities must match exactly")
    if not isinstance(repeats,int) or repeats<2:
        raise ValueError("At least two bootstrap repeats are required")
    keys=sorted(left)
    if any(not isinstance(k,tuple) or len(k)!=2 for k in keys):
        raise ValueError("Keys must be (domain, query_id) pairs")
    differences=np.array([left[k]-right[k] for k in keys],dtype=float)
    if not np.isfinite(differences).all():
        raise ValueError("Metric differences must be finite")
    groups=[np.array([i for i,k in enumerate(keys) if k[0]==d]) for d in sorted({k[0] for k in keys})]
    rng=np.random.default_rng(seed); draws=np.empty(repeats)
    for i in range(repeats):
        if hierarchical:
            selected=rng.integers(len(groups),size=len(groups))
            # Sample queries independently for each occurrence of a repeated domain.
            draws[i]=np.mean([differences[rng.choice(groups[j],size=len(groups[j]),replace=True)].mean()
                              for j in selected])
        else:
            draws[i]=rng.choice(differences,size=len(differences),replace=True).mean()
    estimate=np.mean([differences[g].mean() for g in groups]) if hierarchical else differences.mean()
    return {"difference":float(estimate),"ci95":np.quantile(draws,[.025,.975]).tolist(),
            "repeats":repeats,"seed":seed,"hierarchical":hierarchical}
