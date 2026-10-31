"""Linear source-fitted models, with fully specified query and domain weighting."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.linear_model import LogisticRegression

from .ranking import FEATURES, features, restore_grid


@dataclass
class FusionModel:
    kind: str
    weights: list
    intercept: float = 0.0
    grid_step: float = None
    rrf_constant: float = 60.0

    def __post_init__(self):
        if self.kind not in ("listwise", "tie_listwise", "logistic"):
            raise ValueError("Unknown model kind")
        if len(self.weights) != len(FEATURES) or not np.isfinite(self.weights).all():
            raise ValueError("Model must contain eleven finite weights")
        if not np.isfinite(self.intercept) or not np.isfinite(self.rrf_constant) or self.rrf_constant < 0:
            raise ValueError("Invalid model metadata")
        if self.grid_step is not None and (not np.isfinite(self.grid_step) or self.grid_step <= 0):
            raise ValueError("Invalid grid step")

    def scores(self, query):
        s = restore_grid(query.reranker, self.grid_step)
        z = features(query.bm25, s, self.rrf_constant) @ np.asarray(self.weights) + self.intercept
        return expit(z) if self.kind == "logistic" else z

    def to_dict(self):
        return {"schema_version":1, "feature_names":list(FEATURES),
                "kind":self.kind, "weights":[float(x) for x in self.weights],
                "intercept":float(self.intercept), "grid_step":self.grid_step,
                "rrf_constant":self.rrf_constant}

    @classmethod
    def from_dict(cls, row):
        if row.get("schema_version") != 1 or row.get("feature_names") != list(FEATURES):
            raise ValueError("Model schema or feature order differs")
        return cls(**{k:row[k] for k in ("kind","weights","intercept","grid_step","rrf_constant")})


def _training_lists(queries, domain_equal, grid_step, rrf_constant, columns, tied):
    records = []
    for q in queries:
        s = restore_grid(q.reranker, grid_step)
        x, y = features(q.bm25, s, rrf_constant), q.labels
        if columns is not None:
            x = x.copy()
            x[:, [i for i in range(len(FEATURES)) if i not in columns]] = 0
        groups = [np.flatnonzero(s == value) for value in np.unique(s)] if tied else [np.arange(len(s))]
        groups = [g for g in groups if y[g].sum()>0 and (not tied or len(g)>1)]
        if groups:
            records.append((q.domain, [(x[g], y[g]/y[g].sum()) for g in groups]))
    if not records:
        raise ValueError("No eligible positive-containing training lists")
    counts = Counter(d for d,_ in records)
    query_weights = np.array([1/counts[d] if domain_equal else 1 for d,_ in records], dtype=float)
    query_weights /= query_weights.sum()
    lists = [(x,t,float(a)/len(groups)) for (_,groups),a in zip(records,query_weights) for x,t in groups]
    return lists, len(records)


def _loss_gradient(w, lists, penalty):
    value, gradient = .5*penalty*np.dot(w,w), penalty*w.copy()
    for x,t,a in lists:
        z=x@w
        lse=logsumexp(z)
        value += a*(lse-t@z)
        gradient += a*x.T@(np.exp(z-lse)-t)
    return float(value), gradient


def fit_listfuse(queries, domain_equal=False, grid_step=None, rrf_constant=60,
                 regularization=1.0, columns=None, tie_restricted=False):
    queries = list(queries)
    if not np.isfinite(regularization) or regularization<0:
        raise ValueError("regularization must be finite and nonnegative")
    if columns is not None and (not columns or any(i not in range(11) for i in columns)):
        raise ValueError("columns must contain valid zero-based feature indices")
    lists,n = _training_lists(queries,domain_equal,grid_step,rrf_constant,columns,tie_restricted)
    result=minimize(_loss_gradient,np.zeros(11),args=(lists,regularization/n),
                    jac=True,method="L-BFGS-B",options={"maxiter":500,"ftol":1e-12,"gtol":1e-8})
    if not result.success:
        raise RuntimeError(f"Listwise optimization failed: {result.message}")
    return FusionModel("tie_listwise" if tie_restricted else "listwise",result.x.tolist(),
                       grid_step=grid_step,rrf_constant=rrf_constant)


def fit_logistic(queries, grid_step=None, rrf_constant=60, c=1.0):
    xs,ys,ws=[],[],[]
    if not np.isfinite(c) or c<=0:
        raise ValueError("c must be positive and finite")
    for q in queries:
        y=q.labels; positives=y.sum(); negatives=len(y)-positives
        if positives and negatives:
            xs.append(features(q.bm25,restore_grid(q.reranker,grid_step),rrf_constant))
            ys.append(y); ws.append(np.where(y>0,.5/positives,.5/negatives))
    if not xs:
        raise ValueError("Logistic training needs lists containing both classes")
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error",ConvergenceWarning)
        fit=LogisticRegression(C=c,solver="lbfgs",max_iter=1000).fit(
            np.concatenate(xs),np.concatenate(ys),sample_weight=np.concatenate(ws))
    return FusionModel("logistic",fit.coef_[0].tolist(),float(fit.intercept_[0]),grid_step,rrf_constant)


def select_alpha(queries, grid=None, seeds=range(20), grid_step=None):
    from .ranking import rank
    from .evaluation import evaluate, macro_average
    queries, seeds = list(queries), tuple(seeds)
    grid = np.linspace(0,1,21) if grid is None else np.asarray(list(grid),dtype=float)
    if not queries or not seeds or grid.size==0 or not np.isfinite(grid).all() or np.any((grid<0)|(grid>1)):
        raise ValueError("Nonempty training data, seeds and a valid alpha grid are required")
    values=[]
    for alpha in grid:
        per_query = [np.mean([evaluate(q,rank(q,"convex",seed,alpha=alpha,grid_step=grid_step))["ndcg@10"]
                              for seed in seeds]) for q in queries]
        values.append(macro_average(queries,per_query))
    best=max(values)
    candidates=[float(a) for a,v in zip(grid,values) if abs(v-best)<=1e-12]
    return min(candidates,key=lambda a:(abs(a-.5),a))
