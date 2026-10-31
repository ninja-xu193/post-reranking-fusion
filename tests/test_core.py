"""All fixtures are synthetic; no study predictions or fitted weights are committed."""
import contextlib
from dataclasses import replace
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.optimize import check_grad

from postfusion import Query,load_queries,features,rank,restore_grid
from postfusion import evaluate,fit_listfuse,fit_logistic,FusionModel,tie_gain_ratio
from postfusion.learning import _training_lists,_loss_gradient,select_alpha
from postfusion.protocol import permute_bm25,leave_one_domain_out
from postfusion.statistics import paired_bootstrap,holm
from postfusion.cli import main

EXAMPLE=Path(__file__).resolve().parents[1]/"examples/synthetic.jsonl"


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.queries=load_queries(EXAMPLE)
        self.q=self.queries[0]

    def test_validation(self):
        for kwargs in [{"doc_ids":("a","a","c")},{"reranker":(float("nan"),.8,.4)},
                       {"qrels":{"a":-1}},{"query_id":None},{"doc_ids":()}]:
            with self.assertRaises(ValueError):
                replace(self.q,**kwargs)

    def test_input_is_frozen(self):
        with self.assertRaises(TypeError):
            self.q.qrels["a"]=4

    def test_average_ranks(self):
        x=features([1,2,3],[.8,.8,.4])
        np.testing.assert_allclose(x[:,3],[1/1.5,1/1.5,1/3])
        self.assertEqual(x.shape,(3,11))
        np.testing.assert_allclose(features([1,1],[0,0])[:,:2],0)

    def test_grid_restore(self):
        x=restore_grid([.7000000000000001,.7,.55],.05)
        self.assertEqual(x[0],x[1])
        with self.assertRaises(ValueError):
            restore_grid([.7123],.05)

    def test_tie_only_rule(self):
        self.assertEqual(rank(self.q,"T"),["b","a","c"])
        self.assertEqual(rank(self.q,"convex",alpha=1e-8),rank(self.q,"T"))
        for seed in range(20):
            self.assertEqual(rank(self.q,"D",seed)[-1],"c")

    def test_candidate_order_does_not_change_tie_priority(self):
        indices=[2,0,1]
        shuffled=replace(self.q,doc_ids=tuple(self.q.doc_ids[i] for i in indices),
                         bm25=tuple(self.q.bm25[i] for i in indices),
                         reranker=tuple(self.q.reranker[i] for i in indices))
        for method in ["D","T","rrf","convex","oracle"]:
            self.assertEqual(rank(self.q,method,19),rank(shuffled,method,19))

    def test_full_judgment_denominator(self):
        metrics=evaluate(self.q,["b","c","a"])
        self.assertAlmostEqual(metrics["map@10"],2/3)
        self.assertAlmostEqual(metrics["recall@5"],2/3)
        self.assertLess(metrics["ndcg@10"],1)
        self.assertEqual(evaluate(self.queries[-1],rank(self.queries[-1]))["ndcg@10"],0)
        with self.assertRaises(ValueError):
            evaluate(self.q,["a","a","c"])

    def test_oracle_is_a_within_tie_upper_bound(self):
        q=Query("toy","test",("a","b","c"),(1,2,3),(.5,.5,.5),{"a":2,"b":1})
        upper=evaluate(q,rank(q,"oracle"))["ndcg@10"]
        for order in itertools.permutations(q.doc_ids):
            self.assertGreaterEqual(upper+1e-12,evaluate(q,order)["ndcg@10"])
        self.assertIsNone(tie_gain_ratio(.5,.5,.5))
        self.assertAlmostEqual(tie_gain_ratio(.2,.4,.6),.5)

    def test_listwise_gradient(self):
        lists,n=_training_lists(self.queries,True,None,60,None,False)
        w=np.linspace(-.3,.3,11)
        error=check_grad(lambda z:_loss_gradient(z,lists,1/n)[0],
                         lambda z:_loss_gradient(z,lists,1/n)[1],w)
        self.assertLess(error,1e-6)

    def test_fits_and_roundtrip(self):
        for equal in [False,True]:
            model=fit_listfuse(self.queries,domain_equal=equal)
            recovered=FusionModel.from_dict(model.to_dict())
            np.testing.assert_allclose(model.scores(self.q),recovered.scores(self.q))
            self.assertEqual(len(rank(self.q,"listfuse",model=model)),3)
        lr=fit_logistic(self.queries)
        self.assertTrue(np.isfinite(lr.scores(self.q)).all())

    def test_restricted_and_single_signal_controls(self):
        ltb=fit_listfuse(self.queries,tie_restricted=True)
        self.assertEqual(rank(self.q,"ltb",model=ltb)[-1],"c")
        full=fit_listfuse(self.queries)
        self.assertEqual(rank(self.q,"listfuse",model=full,restrict_ties=True)[-1],"c")
        model=fit_listfuse(self.queries,columns=[1,3,5])
        self.assertEqual(model.weights[0],0)

    def test_permutation(self):
        p=permute_bm25(self.q,21)
        self.assertEqual(sorted(p.bm25),sorted(self.q.bm25))
        self.assertEqual(p.bm25,permute_bm25(self.q,21).bm25)
        self.assertEqual(p.qrels,self.q.qrels)
        changed_labels=replace(self.q,qrels={"a":1})
        self.assertEqual(p.bm25,permute_bm25(changed_labels,21).bm25)

    def test_source_holdout_and_alpha(self):
        rows=leave_one_domain_out(self.queries,seeds=[0,1])
        self.assertEqual(len(rows),len(self.queries))
        self.assertIn(select_alpha(self.queries,grid=[0,.5,1],seeds=[0]),[0,.5,1])

    def test_statistics(self):
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
        left={("a","1"):.5,("b","1"):.8}
        right={("a","1"):.4,("b","1"):.7}
        result=paired_bootstrap(left,right,repeats=50)
        self.assertAlmostEqual(result["difference"],.1)
        np.testing.assert_allclose(result["ci95"],[.1,.1])
        with self.assertRaises(ValueError):
            paired_bootstrap(left,{("a","2"):.4},repeats=5)

    def test_cli_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"model.json"
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["fit","--train",str(EXAMPLE),"--output",str(output)]),0)
                self.assertEqual(main(["fit","--train",str(EXAMPLE),"--output",str(output)]),2)
                self.assertEqual(main(["evaluate","--input",str(EXAMPLE),"--method","listfuse","--model",str(output)]),0)
                self.assertEqual(main(["evaluate","--input",str(EXAMPLE),"--seeds","0"]),2)


if __name__=="__main__":
    unittest.main()
