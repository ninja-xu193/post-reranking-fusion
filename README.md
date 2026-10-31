# Post-reranking fusion

Reference code for comparing lexical tie-breaking and joint score fusion on fixed candidate lists. Inference is outside this package: each input candidate already has a retrieval score and a reranker score.

This repository contains methods, tests and a small **synthetic** example. It does not distribute research results, benchmark queries, prediction caches, trained coefficients, model checkpoints or paper files. It is a reference implementation, not a complete experimental reproduction bundle.

[中文说明](README.zh-CN.md)

## Install

Python 3.9 or newer is required; Python 3.11 is recommended for a new environment. No GPU is needed.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
```

On Windows, activate with `.venv\Scripts\activate`. Runtime dependencies are NumPy, SciPy and scikit-learn. `requirements-tested.txt` records one locally tested numerical environment. This is the CPU analysis environment, not a reranker inference environment.

## Quick start

```sh
postfusion evaluate --input examples/synthetic.jsonl --method D
postfusion evaluate --input examples/synthetic.jsonl --method T
postfusion evaluate --input examples/synthetic.jsonl --method convex --alpha 0.25
postfusion fit --train examples/synthetic.jsonl --output outputs/toy-model.json
postfusion evaluate --input examples/synthetic.jsonl --method listfuse --model outputs/toy-model.json
postfusion evaluate --input examples/synthetic.jsonl --method listfuse --model outputs/toy-model.json --restrict-ties
```

These commands demonstrate the API on invented data. Training and evaluating on the same toy file is not an effectiveness experiment. Use separate source and test files for a real study. The CLI writes fitted coefficients only when an explicit output path is supplied and refuses to overwrite existing files. Evaluation is printed to standard output; it is not committed to the repository.

## Ranking rules

| Rule | Final ordering |
|---|---|
| `D` | Reranker score, then common random priority |
| `T` | Reranker score, BM25 score, then common priority |
| `rrf` | Reciprocal rank fusion with average feature ranks |
| `convex` | Query-normalized `(1-alpha) * reranker + alpha * BM25` |
| `logistic` | Pointwise logistic linear decision value over shared features |
| `listfuse` | Linear listwise score over shared features |
| `ltb` | Reranker score first, separately trained tie-listwise score second |
| `oracle` | Reranker score first, judgment grade second; evaluation only |

`--restrict-ties` applies the **same fitted** joint score within reranker ties. It differs from separately fitting LTB. Neither comparison identifies a universal causal share of improvement.

All rules use deterministic per-document priorities keyed by seed, domain, query and document ID. The priorities are shared across methods and independent of input order and relevance. The implementation uses 20 priorities by default. These keys are explicitly specified here and need not reproduce the incidental order of a historical run.

Use `--grid-step 0.05` only when the scoring interface is known to produce that grid. Off-grid scores fail validation rather than being silently quantized. Probability scores should normally retain full precision. Fitted models store their grid and feature policy.

## Methods and protocol

See [method definitions](docs/methods.md) for all eleven feature columns, objectives, tie-list weighting, metric denominators and statistical assumptions. [Input format](docs/input-format.md) describes the JSONL schema.

```python
from postfusion import load_queries, fit_listfuse
from postfusion.protocol import leave_one_domain_out, permute_bm25

queries = load_queries("examples/synthetic.jsonl")
held_out_metrics = leave_one_domain_out(queries, method="listfuse", seeds=range(20))
model = fit_listfuse(queries, domain_equal=True)
permuted_queries = [permute_bm25(q, seed=17) for q in queries]
```

Permutation is within a query. Passing the new queries through the feature constructor recomputes lexical ranks, normalization and interactions. No original lexical rank is carried over.

The code uses stable log-softmax cross-entropy and fractional average ranks. It is a cleaned reference implementation, not a promise of bit-for-bit agreement with development scripts. In particular, LTB's exact weighting is defined here. Reproducing a particular reported table also requires that run's inputs, split, score adapter, seeds and fitted parameters, none of which are included.

## Data and privacy

The examples contain only invented document identifiers and scores. Add your own datasets outside the checkout or in an ignored local directory. The package does not download models or datasets and does not contact a hosted inference service.

Before preparing an upload, run:

```sh
python tools/check_release.py
```

The release check uses an allowlist, rejects unexpected data/binary files and checks for common credentials, machine paths and internal addresses. It is a precaution, not proof that arbitrary future files are safe. Review the file list before every release. See [release scope](docs/release-scope.md).

## License

MIT for this code. Dataset and model licenses are separate and are not granted by this repository. No third-party model implementation or weight file is bundled.

## References

- Robertson and Zaragoza. The Probabilistic Relevance Framework: BM25 and Beyond. [DOI](https://doi.org/10.1561/1500000019).
- Cormack, Clarke and Buettcher. Reciprocal rank fusion outperforms condorcet and individual rank learning methods. [DOI](https://doi.org/10.1145/1571941.1572114).
- Cao et al. Learning to Rank: From Pairwise Approach to Listwise Approach. [DOI](https://doi.org/10.1145/1273496.1273513).
- McSherry and Najork. Computing Information Retrieval Performance Measures Efficiently in the Presence of Tied Scores. [DOI](https://doi.org/10.1007/978-3-540-78646-7_38).
