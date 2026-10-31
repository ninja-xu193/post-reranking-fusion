# Method definitions

## Inputs and features

Normalization is min--max within each query and signal; constant signals map to zero. Ranks descend by score and use the mean occupied position for ties. Fractional average ranks enter reciprocal features directly.

| Index | Feature |
|---|---|
| 1 | Normalized retrieval score |
| 2 | Normalized reranker score |
| 3, 4 | Reciprocal retrieval and reranker ranks |
| 5, 6 | Retrieval and reranker ranks divided by candidate count |
| 7 | Signed rank difference divided by candidate count |
| 8 | Absolute rank difference divided by candidate count |
| 9 | Product of normalized scores |
| 10 | Difference of normalized scores |
| 11 | Query-normalized RRF score, constant 60 |

Columns 7 and 10 are redundant linear combinations. Retaining them changes the regularized parameterization, not the independent linear information. Feature-subset controls are available through zero-based `columns` in `fit_listfuse`.

## Listwise fitting

A linear score `z = x @ w` has a within-list softmax. Binary positive labels are normalized to sum to one. The loss uses log-sum-exp for numerical stability. Queries without a relevant candidate do not enter fitting but remain in evaluation.

Ordinary fitting averages eligible queries. Domain-equal fitting gives equal total weight to each source domain with eligible queries and averages eligible queries within it. The penalty is `regularization * ||w||^2 / (2 * N)`, with `N` the eligible query count and default regularization 1. No intercept is fitted. L-BFGS-B starts at zero with 500 iterations, function tolerance `1e-12` and gradient tolerance `1e-8`. Optimization failure is reported.

## Learned tie-breaking

Tie-listwise fitting splits each source query into exact reranker-score groups. It retains groups of at least two documents with at least one binary-positive label. All-positive groups are retained. Losses are averaged over eligible groups within a query, then queries using the ordinary or domain-equal rule above. The penalty denominator is the number of eligible queries, not the group count. Features are constructed from the full query before group splitting.

Evaluation ranks by reranker score first and the fitted score second. This definition makes the cleaned reference unambiguous. It is not a claim that an unavailable historical LTB implementation used identical weighting.

## Logistic and convex fitting

Logistic fitting uses the same eleven columns, `C=1`, L-BFGS and at most 1,000 iterations. Within each eligible query, positives have weights `0.5 / positive_count` and negatives `0.5 / negative_count`. Queries missing either class are excluded. Ranking uses predicted probabilities. Convergence warnings are treated as errors.

Convex fusion assigns BM25 weight `alpha`. Source-only selection maximizes domain-equal nDCG@10 over a 21-point grid from zero to one. Ties select the weight closest to 0.5, then the smaller value. Evaluation priorities are averaged before weight selection. Do not choose weights on the held-out target set.

## Evaluation

Ranked IDs are explicit, preventing a downstream evaluator from silently choosing a second tie order. nDCG@10 uses linear judgment gains and an ideal list from all query judgments. AP@10 divides by the **total** judged relevant count, not `min(R, 10)`. Recall@5 uses the same total-relevant denominator. A query with no relevant judgment returns zero. Library metrics are on a 0--1 scale; multiply differences by 100 for percentage points.

The oracle orders each reranker tie by judgment grade. It is an evaluation upper bound, not a deployable method. `tie_gain_ratio` returns `None` when the oracle-minus-direct denominator is nonpositive.

## Statistical scope

`paired_bootstrap` accepts two maps keyed by the same `(domain, query_id)` pairs. Hierarchical draws resample domains and independently resample queries within **each occurrence** of a sampled domain. Nonhierarchical draws resample query pairs. Seed repeats should be averaged inside each query before calling it. Intervals condition on the supplied predictions and do not include refitting.

`holm` adjusts exactly the p-values supplied by the caller. It does not choose a comparison family, perform a hypothesis test or make nominal intervals simultaneous. Define comparison families before interpreting results. Equality of rounded means and nonsignificance are not equivalence tests.
