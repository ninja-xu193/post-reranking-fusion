"""No downloads, GPU calls or implicit writes; all inputs are user supplied."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

from .data import load_queries
from .ranking import rank
from .learning import FusionModel, fit_listfuse, fit_logistic
from .evaluation import evaluate, macro_average


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    fit=commands.add_parser("fit",help="Fit a model on source queries only")
    fit.add_argument("--train",required=True)
    fit.add_argument("--output",required=True,help="New local model JSON; not a model checkpoint")
    fit.add_argument("--kind",choices=["listwise","tie_listwise","logistic"],default="listwise")
    fit.add_argument("--domain-equal",action="store_true")
    fit.add_argument("--grid-step",type=float)
    ev=commands.add_parser("evaluate",help="Print seed-averaged query and domain metrics")
    ev.add_argument("--input",required=True)
    ev.add_argument("--method",choices=["D","T","bm25","rrf","convex","listfuse","logistic","ltb","oracle"],default="T")
    ev.add_argument("--model")
    ev.add_argument("--alpha",type=float,default=.25)
    ev.add_argument("--grid-step",type=float)
    ev.add_argument("--seeds",type=int,default=20,help="Seed count, starting at zero")
    ev.add_argument("--restrict-ties",action="store_true")
    args=parser.parse_args(argv)
    try:
        if args.command=="fit":
            path=Path(args.output)
            if path.exists():
                raise ValueError("Output already exists; choose a new local filename")
            queries=load_queries(args.train)
            if args.kind=="logistic":
                if args.domain_equal:
                    raise ValueError("domain-equal is available for listwise objectives only")
                model=fit_logistic(queries,grid_step=args.grid_step)
            else:
                model=fit_listfuse(queries,domain_equal=args.domain_equal,grid_step=args.grid_step,
                                  tie_restricted=args.kind=="tie_listwise")
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open("x",encoding="utf-8") as stream:
                json.dump(model.to_dict(),stream,indent=2,allow_nan=False)
            print("Source-fitted model saved locally.")
        else:
            if args.seeds<1:
                raise ValueError("seeds must be positive")
            queries=load_queries(args.input)
            model=FusionModel.from_dict(json.loads(Path(args.model).read_text())) if args.model else None
            records=[]
            for q in queries:
                repeats=[evaluate(q,rank(q,args.method,seed,model,args.alpha,args.grid_step,
                                         restrict_ties=args.restrict_ties)) for seed in range(args.seeds)]
                metrics={k:float(np.mean([v[k] for v in repeats])) for k in repeats[0]}
                records.append({"domain":q.domain,"query_id":q.query_id,"metrics":metrics})
            macro={k:macro_average(queries,[r["metrics"][k] for r in records]) for k in records[0]["metrics"]}
            print(json.dumps({"scale":"0--1","method":args.method,"seeds":args.seeds,
                              "domain_equal_macro":macro,"queries":records},indent=2,allow_nan=False))
        return 0
    except (OSError,ValueError,RuntimeError) as exc:
        print(f"Error: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
